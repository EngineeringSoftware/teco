import base64
import datetime
import traceback
from pathlib import Path
from typing import List, Tuple, Union
from urllib.error import HTTPError
from urllib.request import urlopen

import seutil as su
import spdx_lookup as lookup
from github import UnknownObjectException
from jsonargparse import CLI
from jsonargparse.typing import Path_dc, Path_drw, Path_fr
from seutil.project import Project
from teco.data.maven import MavenModule, MavenProjectHelper
from teco.data.tool import ensure_tool_versions
from teco.macros import Macros
from tqdm import tqdm

logger = su.log.get_logger(__name__, su.log.INFO)


class ReposCollector:
    """
    Collection and filtering of repositories.
    """

    TAG_TIME_LIMIT = 1577836800  # 2020-01-01 00:00 UTC
    CLONE_TIMEOUT = 300
    COMPILE_TIMEOUT = 300
    TEST_TIMEOUT = 1800

    def __init__(
        self, downloads_dir: Union[Path_drw, Path_dc, Path] = Macros.downloads_dir
    ):
        if not isinstance(downloads_dir, Path):
            downloads_dir = Path(downloads_dir.abs_path)
        self.downloads_dir = downloads_dir

    def search_repos_from_lists(self, lists: List[Path_fr], out_dir: Path_dc):
        lists = [Path(l.abs_path) for l in lists]
        out_dir = Path(out_dir.abs_path)
        su.io.mkdir(out_dir, fresh=True)

        repo2source = {}
        for l in lists:
            for repo_name in su.io.load(l, su.io.Fmt.txtList):
                if repo_name in repo2source:
                    logger.warning(
                        f"{repo_name} already in {repo2source[repo_name]}, appear again in {l.stem}"
                    )
                repo2source[repo_name] = l.stem

        print(f"In total {len(repo2source)} repos to search")

        logs = {
            "start_time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "success_projects": [],
            "fail_projects": [],
        }

        success = 0
        fail = 0
        projects = []
        pbar = tqdm(desc=f"Processing... (+{success} -{fail})", total=len(repo2source))
        for full_name_raw, sources in repo2source.items():
            pbar.set_description(f"Processing {full_name_raw} (+{success} -{fail})")
            try:
                # put together a project instance, with the help of GitHub API
                # TODO: trying to access a deleted repo now throws UnknownObjectException ("Not Found") rather than ValidationError, which incorrectly triggers the retry logic (but this should probably be fixed upstream in seutil)
                repo = su.GitHubUtils.ensure_github_api_call(
                    lambda g: g.get_repo(full_name_raw), max_retry_times=2
                )
                url = su.GitHubUtils.ensure_github_api_call(lambda g: repo.clone_url)
                # get the latest user/repo name from GitHub API
                user_name = su.GitHubUtils.ensure_github_api_call(
                    lambda g: repo.owner.login, max_retry_times=2
                )
                repo_name = su.GitHubUtils.ensure_github_api_call(
                    lambda g: repo.name, max_retry_times=2
                )
                full_name = f"{user_name}_{repo_name}"
                p = Project(full_name=full_name, url=url)
                p.data["repo"] = repo_name
                p.data["user"] = user_name
                p.data["branch"] = su.GitHubUtils.ensure_github_api_call(
                    lambda g: repo.default_branch, max_retry_times=2
                )
                p.data["stars"] = su.GitHubUtils.ensure_github_api_call(
                    lambda g: repo.stargazers_count, max_retry_times=2
                )
                p.data["sources"] = sources

                # Try to download the project; remove previously downloaded repo
                with su.TimeUtils.time_limit(self.CLONE_TIMEOUT):
                    p.clone(self.downloads_dir)

                # Select a sha: either a tag (approximation of release) after 2020.1.1, or the latest commit
                p.checkout(p.data["branch"], forced=True)
                with su.io.cd(p.dir):
                    rr = su.bash.run(
                        "git describe --tags $(git rev-list --tags --max-count=1)"
                    )
                    if rr.returncode == 0:
                        tag = rr.stdout.strip()
                        tag_time = int(
                            su.bash.run(
                                f"git log {tag} -1 --pretty='%at'", 0
                            ).stdout.strip()
                        )
                        if tag_time >= self.TAG_TIME_LIMIT:
                            p.checkout(tag, forced=True)
                            p.data["tag"] = tag
                            p.data["tag_time"] = tag_time

                p.data["sha"] = p.get_cur_revision()

                # Put in the project
                projects.append(p)
                logs["success_projects"].append(p.full_name)
                success += 1
            except KeyboardInterrupt:
                input(
                    "\n***User interrupted*** Press Ctrl-C again to abort. Press ENTER to skip current project and continue..."
                )
                logger.warning(f"Project {repo_name} failed: User interrupted")
                logs["fail_projects"].append((repo_name, "User interrupted"))
                fail += 1
            except BaseException:
                logger.warning(f"Project {repo_name} failed: {traceback.format_exc()}")
                logs["fail_projects"].append((repo_name, traceback.format_exc()))
                fail += 1
            finally:
                logger.info(f"Finish processing {repo_name}")
                pbar.update(1)

                # free up disk space
                p.remove()
        pbar.set_description(f"Finished (+{success} -{fail})")
        pbar.close()
        logs["finish_time"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"

        su.io.dump(out_dir / "repos.json", projects, su.io.Fmt.jsonPretty)
        su.io.dump(out_dir / "logs.json", logs, su.io.Fmt.jsonNoSort)

    allowed_licenses = [
        "Apache License 2.0",
        'BSD 2-clause "Simplified" License',
        'BSD 3-clause "New" or "Revised" License',
        "Common Public License 1.0",
        "Creative Commons Attribution Share Alike 4.0",
        "Creative Commons Zero v1.0 Universal",
        "GNU Affero General Public License v3.0",
        "GNU General Public License v2.0 only",
        "GNU General Public License v3.0 only",
        "GNU Lesser General Public License v2.1 only",
        "GNU Lesser General Public License v3.0 only",
        "ISC License",
        "MIT License",
        "Mozilla Public License 2.0",
    ]

    def filter_licensed_maven_compilable(self, repos_file: Path_fr, out_dir: Path_dc):
        repos_file = Path(repos_file.abs_path)
        out_dir = Path(out_dir.abs_path)

        su.io.mkdir(out_dir, fresh=True)

        # check tool versions
        ensure_tool_versions()

        projects: List[Project] = su.io.load(repos_file, clz=List[Project])
        filtered_projects: List[Project] = []
        logs = {
            "start_time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "maven_version": su.bash.run("mvn -v").stdout.strip(),
            "maven_projects_single_module": [],
            "maven_projects_multi_module": [],
            "bad_license": [],
            "non_maven_projects": [],
            "success_projects": [],
            "fail_projects": [],
        }
        success = 0
        ignored = 0
        fail = 0
        pbar = tqdm(
            desc=f"Processing... (S{success} I{ignored} F{fail})", total=len(projects)
        )

        license_dir = out_dir / "licenses"
        su.io.mkdir(license_dir)

        for p in projects:
            pbar.set_description(
                f"Processing {p.full_name} (S{success} I{ignored} F{fail})"
            )
            status = "fail"
            try:
                # check license of the project
                license_type, license_text = self.get_project_license(p)
                if license_type in self.allowed_licenses:
                    p.data["license_type"] = license_type

                    # save license full text
                    su.io.dump(license_dir / f"{p.full_name}.txt", license_text)
                else:
                    logs["bad_license"].append(p.full_name)
                    status = "ignored"
                    continue

                # detect if the project is Maven by try to access pom.xml
                try:
                    pom_xml_url = (
                        (p.url[: -len(".git")] if ".git" in p.url else p.url)
                        + "/blob/"
                        + p.data["sha"]
                        + "/pom.xml"
                    )
                    urlopen(pom_xml_url)
                    p.data["build_system"] = "mvn"
                except HTTPError:
                    logs["non_maven_projects"].append(p.full_name)
                    status = "ignored"
                    continue

                # download the project and checkout to pre-selected version
                with su.TimeUtils.time_limit(self.CLONE_TIMEOUT):
                    p.clone(self.downloads_dir)
                p.checkout(p.data["sha"], forced=True)

                with su.io.cd(p.dir):
                    # figure out if the project is multi-module
                    rr = su.bash.run(
                        "mvn -Dexec.executable='bash' -Dexec.args='-c '\"'\"'echo ${project.groupId}:${project.artifactId} ${project.packaging} ${PWD}'\"'\"'' exec:exec -q"
                    )
                    if rr.returncode != 0:
                        raise RuntimeError(
                            "Running maven command (for getting modules) failed"
                        )

                    all_modules: List[MavenModule] = [
                        MavenModule.parse_from_output(l, p.dir)
                        for l in rr.stdout.splitlines()
                    ]
                    if len(all_modules) == 1:
                        # single module
                        p.data["mvn_multi_module"] = False
                        logs["maven_projects_single_module"].append(p.full_name)
                    else:
                        # multi module
                        p.data["mvn_multi_module"] = True
                        logs["maven_projects_multi_module"].append(p.full_name)

                        # only keep non-parent & jar modules
                        modules = [
                            m
                            for m in all_modules
                            if m.rel_path != "." and m.packaging == "jar"
                        ]

                        if len(modules) == 0:
                            raise RuntimeError(
                                "No usable module in a multi-module Maven project"
                            )

                        p.data["mvn_modules"] = [m.rel_path for m in modules]

                    # estimate number of tests, ignore the projects without any test
                    estimate_num_test_method = (
                        MavenProjectHelper.estimate_num_test_method(p)
                    )
                    if estimate_num_test_method == 0:
                        raise RuntimeError("No test method detected")
                    p.data["estimate_num_test_method"] = estimate_num_test_method

                    # try to test-compile
                    with su.TimeUtils.time_limit(self.COMPILE_TIMEOUT):
                        su.bash.run(
                            "mvn clean test-compile", 0, timeout=self.COMPILE_TIMEOUT
                        )

                    # if everything passes until this point, success
                    filtered_projects.append(p)
                    logs["success_projects"].append(p.full_name)
                    status = "success"
            except KeyboardInterrupt:
                input(
                    "\n***User interrupted*** Press Ctrl-C again to abort. Press ENTER to skip current project and continue..."
                )
                logger.info(f"Project {p.full_name} failed: User interrupted")
                logs["fail_projects"].append((p.full_name, "User interrupted"))
                status = "fail"
            except BaseException:
                logger.info(f"Project {p.full_name} failed: {traceback.format_exc()}")
                logs["fail_projects"].append((p.full_name, traceback.format_exc()))
                status = "fail"
            finally:
                if status == "success":
                    success += 1
                elif status == "ignored":
                    ignored += 1
                    p.remove()
                else:
                    fail += 1
                    p.remove()
                pbar.update(1)

        pbar.set_description(f"Finished (S{success} I{ignored} F{fail})")
        pbar.close()
        logs["finish_time"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"

        su.io.dump(out_dir / "repos.json", filtered_projects, su.io.Fmt.jsonPretty)
        su.io.dump(out_dir / "logs.json", logs, su.io.Fmt.jsonNoSort)

    @classmethod
    def get_project_license(cls, p: Project) -> Tuple[str, str]:
        """
        Gets the license of the project.
        :param p: the Project instance.
        :return: the project's license type and full text.
        """

        conn = su.GitHubUtils.get_github()

        slug = p.data["user"] + "/" + p.data["repo"]
        try:
            repo = conn.get_repo(slug)
            license_text = base64.b64decode(
                repo.get_license().content.encode()
            ).decode()
            license_info = lookup.match(license_text)
            if license_info is not None:
                license_type = str(license_info.license)
            else:
                license_type = "Unknown"
        except UnknownObjectException:
            license_type = "Unknown"
            license_text = None

        return license_type, license_text


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.WARNING)
    CLI(ReposCollector, as_positional=False)
