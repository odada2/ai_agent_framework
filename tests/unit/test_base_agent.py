import unittest
from unittest.mock import MagicMock, patch
import asyncio

from ai_agent_framework.core.llm.base import BaseLLM
from ai_agent_framework.agents.base_agent import BaseAgent