#!/usr/bin/env python
import jinja2
import tomllib

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("."),
    trim_blocks=True,
    lstrip_blocks=True,
)

template = env.get_template("meta.yaml.j2")

output = template.render(pyproject=pyproject)

with open("meta.yaml", "w") as f:
    f.write(output)
