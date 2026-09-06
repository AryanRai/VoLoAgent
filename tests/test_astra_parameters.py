"""Azure deployment names are unprefixed; Astra rejects temperature/max_tokens."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from vlm_orchestrator.vlm.api import chat_create


class AstraParameterTests(unittest.TestCase):
    def test_astra_bare_prefixed_and_snapshot(self):
        for model in ['gpt-6-astra', 'openai/gpt-6-astra', 'gpt-6-astra-2026-09-03']:
            with self.subTest(model=model):
                create = Mock()
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                chat_create(client, model=model, messages=[], temperature=0, max_tokens=512)
                kwargs = create.call_args.kwargs
                self.assertNotIn('temperature', kwargs)
                self.assertNotIn('max_tokens', kwargs)
                self.assertEqual(kwargs['max_completion_tokens'], 2048)
                self.assertEqual(kwargs['reasoning_effort'], 'low')

    def test_claude_46_keeps_original_parameters(self):
        create = Mock()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        chat_create(client, model='anthropic/claude-opus-4.6', messages=[], temperature=0, max_tokens=512)
        self.assertEqual(create.call_args.kwargs['temperature'], 0)
        self.assertEqual(create.call_args.kwargs['max_tokens'], 512)
        self.assertNotIn('reasoning_effort', create.call_args.kwargs)
