from pathlib import Path

from jinja2 import Environment, FileSystemLoader

jinja_environment = Environment(loader=FileSystemLoader(Path("src/prompt_templates/")))


def render_template(template_name: str, **kwargs) -> str:
    """
    Renders a Jinja2 template with the given arguments.
    """
    template = jinja_environment.get_template(template_name)
    try:
        return template.render(**kwargs)
    except Exception as e:
        print(e)
        return ""
