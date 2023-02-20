# TeCo

TeCo is a deep learning model using code semantics to automatically complete the next statement in a test method. Completing tests requires reasoning about the execution of the code under test, which is hard to do with only syntax-level data that existing code completion models use. To solve this problem, we leverage the fact that tests are readily executable. TeCo extracts and uses execution-guided code semantics as inputs for the ML model, and performs reranking via test execution to improve the outputs. On a large dataset with 131K tests from 1270 open-source Java projects, TeCo outperforms the state-of-the-art by 29% in terms of test completion accuracy.

We will release the code and data for TeCo soon.
