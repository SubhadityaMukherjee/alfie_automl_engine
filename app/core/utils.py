import ollama
from jinja2 import Environment


def render_template(jinja_environment:Environment, template_name: str, **kwargs) -> str:
    """
    Renders a Jinja2 template with the given arguments.
    """
    template = jinja_environment.get_template(template_name)
    try:
        return template.render(**kwargs)
    except Exception as e:
        print(e)
        return ""

