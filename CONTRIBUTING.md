# Contributing

Start with the [software quickstart](README.md#try-the-software-path) and
[design documents](docs/README.md). Keep changes focused and comments short and
lowercase; preserve the spelling of identifiers and tool directives.

For software changes, run the relevant Python or C++ tests and record the exact
command and outcome. Changes to model support should include an independent
reference comparison and a rejection case for unsupported input. Do not present
older results as validation of a new revision.

For hardware changes, run the local Verilator testbenches with `make -C tb all`.
Record which tests passed and include a reproducer for any failure.

When reporting a bug, include the revision, tool versions, smallest reproducer,
expected result and actual result. Avoid attaching checkpoints, private data or
large generated build files.

The diagrams include draw.io files and SVG previews. To regenerate both:

```sh
python3 scripts/render_docs_diagrams.py
```

Edit the generator, regenerate the diagrams, and inspect their layout before
submitting changes. See [diagram notes](docs/assets/README.md).

Project code and documentation use the [MIT license](LICENSE). External models,
datasets and dependencies retain their own licenses.
