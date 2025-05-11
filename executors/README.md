# Testing the images
The images need to be built first.  The paths are relative to this file's directory (`./executors` from repo root).

Test all images (defined in the test command):
```
../scripts/test_executors.sh test-all
```

Test a single image:
```
../scripts/test_executors.sh test-one ./lua agnostics-lua-executor
```

Inputs with which the image is tested are defined in `$image_dir/tests`.
The dir names are meaningful and hardcoded, they identify what should be in the `result` field.
A dir can be missing. It makes sense to only start with `$image_dir/tests/success`.

# Executor container protocol
See `./lua/workdir-template/test_harness.py` as a reference for the protocol.

ATTW the protocol must be rigorously followed, because the Python code doesn't allow any deviations.
See [this section](.#protocol-handling-in-python).

The container receives data about the test to execute through the stdin, serialized as a JSONL stream.
Each object is like a single request to execute some tests.

The top object has three fields at the moment: `code`, `test_cases`, and `timeout_s`.
The `test_cases` field holds an array of objects with 2 fields: `input` and `output`.

The container should test the code on the given test cases.
Any test case taking more than `timeout_s` is considered a timeout.
After running the test, the results are returned on stdout as a JSONL stream,
1 object per 1 object on the stdin.

# Adding an image
Copy one of the existing image directories, `lua` is a good starting point.

You'll want to change:
1. The container tag in `build.sh`
2. The base image in `Dockerfile`
3. The Python files in `workdir-template/*`, which are responsible for handling the container I/O.
   - The `test_harness.py` script is used as the container entrypoint.
   - The `container_protocol.py` module contains helper definitions for building responses
     according to the protocol. This file should be the same for All the executors,
     unless we find a very good reason.

Check that all the `container_protocol.py` files are the same with:
```bash
shasum */workdir-template/container_protocol.py
```

After modifying that file for `$LANG`, spread the changes with:
```bash
for d in */; do cp $LANG/workdir-template/container_protocol.py $d/workdir-template; done
```


You can test your image with `../scripts/test_executors.sh test-one` (see section above).
If you can execute the harness script locally, you can also run a command like this one:
```
../scripts/test_executors.sh test-harness -- ./lua python3 ./lua/workdir-template/test_harness.py
```
The arguments after the first one are the entire command to run the harness script.

Finally, you'll also want to modify `def test_all` in `../src/agnostics/cli/test_executors.py`.