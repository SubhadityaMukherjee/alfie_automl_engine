import logging
import os
from typing import Dict, List, Tuple
from jinja2 import Environment
from ollama import Client

import json
from app.core.utils import render_template
from app.automlplus.utils import ImageConverter

logger = logging.getLogger(__name__)

client = Client()


