import logging

from jinja2 import Environment

logger = logging.getLogger(__name__)


def render_template(
    jinja_environment: Environment, template_name: str, **kwargs
) -> str:
    """
    Renders a Jinja2 template with the given arguments.
    """
    template = jinja_environment.get_template(template_name)
    try:
        return template.render(**kwargs)
    except Exception as e:
        logger.error("Template render failed: %s", str(e))
        return ""
