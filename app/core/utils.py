import logging

from jinja2 import Environment

logger = logging.getLogger(__name__)


def render_template(
    jinja_environment: Environment, template_name: str, **kwargs
) -> str:
    """
    Renders a Jinja2 template with the given arguments.
    """
    try:
        template = jinja_environment.get_template(template_name)
    except Exception as e:
        logger.error("Failed to load Jinja2 template '%s': %s", template_name, str(e))
        raise RuntimeError(f"Failed to load template '{template_name}': {e}") from e

    try:
        return template.render(**kwargs)
    except Exception as e:
        logger.error(
            "Failed to render Jinja2 template '%s' with kwargs %s: %s",
            template_name,
            kwargs,
            str(e),
        )
        raise RuntimeError(f"Failed to render template '{template_name}': {e}") from e
