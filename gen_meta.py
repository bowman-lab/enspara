#!/usr/bin/env python
import jinja2
import tomllib

rewrites = {
    "tables": "pytables",
}

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

print(pyproject)

for original in rewrites:
    rewritten = rewrites[original]
    pyproject["project"]["dependencies"] = [ dep.replace(original, rewritten) for dep in pyproject["project"]["dependencies"] ]
    pyproject["build-system"]["requires"] = [ dep.replace(original, rewritten) for dep in pyproject["build-system"]["requires"] ]
    pyproject["project"]["optional-dependencies"] = {
        deptype: [ dep.replace(original, rewritten) for dep in pyproject["project"]["optional-dependencies"][deptype] ]
        for deptype in pyproject["project"]["optional-dependencies"]
    }
    pyproject["build-system"]["requires"].append("pip")

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("."),
    trim_blocks=True,
    lstrip_blocks=True,
)

template = env.get_template("meta.yaml.j2")

output = template.render(pyproject=pyproject)

with open("meta.yaml", "w") as f:
    f.write(output)
