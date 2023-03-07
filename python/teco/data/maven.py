import dataclasses
import os
from pathlib import Path, PurePath
from typing import List, Optional

import seutil as su
from seutil.project import Project

logger = su.log.get_logger(__name__, su.log.INFO)


class MavenProject:
    def __init__(self, project: Project):
        assert project.data["build_system"] == "mvn"
        self.project: Project = project

        # Get multi-module information
        self.multi_module: bool = project.data.get("mvn_multi_module", False)
        self.module: Optional[str] = project.data.get("mvn_module")
        self.module_rel_path: Optional[str] = project.data.get(
            "mvn_module_path", self.module
        )

        # Get maven specific properties or use default values
        self.rel_source_path: str = project.data.get("mvn_source_path", "src/main/java")
        self.rel_test_path: str = project.data.get("mvn_test_path", "src/test/java")

        return

    @property
    def full_name(self) -> str:
        return self.project.full_name

    @property
    def dir(self) -> Path:
        if self.multi_module:
            return Path(str(self.project.dir).rsplit("_", 1)[0])
        else:
            return self.project.dir

    @property
    def module_dir(self) -> Path:
        if self.multi_module:
            return self.dir / self.module_rel_path
        else:
            return self.dir

    @property
    def data(self) -> dict:
        return self.project.data

    @property
    def source_path(self) -> Path:
        return self.module_dir / self.rel_source_path

    @property
    def test_path(self) -> Path:
        return self.module_dir / self.rel_test_path

    @property
    def logging_prefix(self):
        return f"Project {self.full_name}: "


@dataclasses.dataclass
class MavenModule:
    # "." means parent module
    rel_path: str = ""
    maven_index: str = ""
    packaging: str = ""

    @classmethod
    def parse_from_output(cls, s, checkout_dir: Path) -> "MavenModule":
        maven_index, packaging, pwd = s.split(maxsplit=2)
        rel_path = str(Path(pwd).relative_to(checkout_dir))
        return cls(
            rel_path=rel_path,
            maven_index=maven_index,
            packaging=packaging,
        )


class MavenProjectHelper:
    @classmethod
    def check_is_mvn_project(cls, p: Project) -> bool:
        """
        Checks if the project is a properly configured maven project.
        The project should have the following data:
          build_system = mvn
          mvn_multi_project
          mvn_modules, if mvn_multi_module = True
        """
        if p.data.get("build_system") != "mvn":
            return False
        if "mvn_multi_module" not in p.data:
            return False
        if p.data["mvn_multi_module"] is True and "mvn_modules" not in p.data:
            return False
        return True

    @classmethod
    def get_app_class_roots(cls, p: Project) -> List[PurePath]:
        """
        Returns the Maven project's all application class file roots.

        :param p: the Project instance.
        :return: a list of relative paths to the application class file roots.
        """
        if not cls.check_is_mvn_project(p):
            raise RuntimeError("Not a Maven project")

        if p.data["mvn_multi_module"]:
            return [PurePath(m) / "target" / "classes" for m in p.data["mvn_modules"]]
        else:
            return [PurePath("") / "target" / "classes"]

    @classmethod
    def get_test_class_roots(cls, p: Project) -> List[PurePath]:
        """
        Returns the Maven project's all test class file roots.

        :param p: the Project instance.
        :return: a list of relative paths to the test class file roots.
        """
        if not cls.check_is_mvn_project(p):
            raise RuntimeError("Not a Maven project")

        if p.data["mvn_multi_module"]:
            return [
                PurePath(m) / "target" / "test-classes" for m in p.data["mvn_modules"]
            ]
        else:
            return [PurePath("") / "target" / "test-classes"]

    @classmethod
    def get_app_java_roots(cls, p: Project) -> List[PurePath]:
        """
        Returns the Maven project's all application java file roots.

        :param p: the Project instance.
        :return: the list of relative paths to the application java file roots.
        """
        if not cls.check_is_mvn_project(p):
            raise RuntimeError("Not a Maven project")

        if p.data["mvn_multi_module"]:
            return [
                PurePath(m) / "src" / "main" / "java" for m in p.data["mvn_modules"]
            ]
        else:
            return [PurePath("") / "src" / "main" / "java"]

    @classmethod
    def get_test_java_roots(cls, p: Project) -> List[PurePath]:
        """
        Returns the Maven project's all test java file roots.

        :param p: the Project instance.
        :return: the list of relative paths to the test java file roots.
        """
        if not cls.check_is_mvn_project(p):
            raise RuntimeError("Not a Maven project")

        if p.data["mvn_multi_module"]:
            return [
                PurePath(m) / "src" / "test" / "java" for m in p.data["mvn_modules"]
            ]
        else:
            return [PurePath("") / "src" / "test" / "java"]

    @classmethod
    def estimate_num_test_method(cls, p: Project) -> int:
        """
        Estimates the number of JUnit test methods in the Maven project by searching for
        @Test annotations in test java files, without running tests.

        Require the project to be cloned and checkout to the correct location.
        """
        test_java_roots = cls.get_test_java_roots(p)

        num_test_method = 0
        for r in test_java_roots:
            if not (p.dir / r).is_dir():
                continue
            for java_path in (p.dir / r).rglob("*.java"):
                with open(java_path, "r") as f:
                    for line in f.readlines():
                        if "@Test" in line:
                            num_test_method += 1

        return num_test_method

    @classmethod
    def get_app_src_path(cls, p: Project) -> str:
        """
        Gets the application src path of the project, require the project is cloned.
        """
        app_java_roots = cls.get_app_java_roots(p)
        return os.pathsep.join([str(p.dir / r) for r in app_java_roots])

    @classmethod
    def get_test_src_path(cls, p: Project) -> str:
        """
        Gets the test src path of the project, require the project is cloned.
        """
        test_java_roots = cls.get_test_java_roots(p)
        return os.pathsep.join([str(p.dir / r) for r in test_java_roots])

    @classmethod
    def get_app_class_path(cls, p: Project) -> str:
        """
        Gets the application class path of the project, require the project is cloned.
        """
        app_class_roots = cls.get_app_class_roots(p)
        return os.pathsep.join([str(p.dir / r) for r in app_class_roots])

    @classmethod
    def get_test_class_path(cls, p: Project) -> str:
        """
        Gets the test class path of the project, require the project is cloned.
        """
        test_class_roots = cls.get_test_class_roots(p)
        return os.pathsep.join([str(p.dir / r) for r in test_class_roots])

    @classmethod
    def get_dependency_classpath(self, p: Project) -> str:
        """
        Gets the dependency class path of the project, require the project is cloned.
        """
        if p.data["mvn_multi_module"]:
            classpaths = set()
            for m in p.data["mvn_modules"]:
                with su.io.cd(p.dir / m):
                    classpath_file = Path(su.io.mktmp("teco", ".txt"))
                    su.bash.run(
                        f"mvn dependency:build-classpath -Dmdep.outputFile={classpath_file}",
                        0,
                    )
                    classpath = su.io.load(classpath_file, su.io.Fmt.txt).strip()
                    su.io.rm(classpath_file)
                    if len(classpath) > 0:
                        classpaths.update(classpath.split(":"))
            return ":".join(classpaths)
        else:
            with su.io.cd(p.dir):
                classpath_file = Path(su.io.mktmp("teco", ".txt"))
                su.bash.run(
                    f"mvn dependency:build-classpath -Dmdep.outputFile={classpath_file}",
                    0,
                )
                classpath = su.io.load(classpath_file, su.io.Fmt.txt).strip()
                su.io.rm(classpath_file)
                return classpath
