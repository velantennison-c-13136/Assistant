import unittest
from unittest.mock import call, patch

from langchain_core.messages import HumanMessage, ToolMessage

import langgraph_agent as agent


class WebResearchTests(unittest.TestCase):
    def setUp(self):
        clarification = patch.object(agent, "_intent_clarification", return_value="")
        clarification.start()
        self.addCleanup(clarification.stop)
        self.messages = [HumanMessage(
            content="Translated question",
            additional_kwargs={"original_request": "Original transliterated question"},
        )]
        self.endpoint = "http://127.0.0.1:11434"
        self.url = "https://example.org/leadership"
        self.evidence = self.messages + [ToolMessage(
            content=f"URL: {self.url}\nPage content: The current leader is Person B.",
            name="fetch_web_page", tool_call_id="page-1",
        )]

    def test_invalid_decision_retries_instead_of_skipping_search(self):
        with patch.object(agent, "llm_text", side_effect=[
            '{"needs_web_search":"false"}',
            '{"needs_web_search":true,"search_query":"current officeholder"}',
        ]) as invoke, patch.object(agent, "emit"):
            self.assertEqual(agent._web_research_query(self.messages, "model", self.endpoint).query, "current officeholder")
        self.assertEqual(invoke.call_count, 2)
        first_prompt = invoke.call_args_list[0].args[2][1].content
        self.assertIn("Original transliterated question", first_prompt)

    def test_missing_decision_fails_explicitly_after_retry(self):
        with patch.object(agent, "llm_text", return_value="{}"), patch.object(agent, "emit"):
            with self.assertRaisesRegex(ValueError, "web search decision: invalid response"):
                agent._web_research_query(self.messages, "model", self.endpoint)

    def test_missing_search_query_retries(self):
        with patch.object(agent, "llm_text", side_effect=[
            '{"needs_web_search":true}',
            '{"needs_web_search":true,"search_query":"current officeholder"}',
        ]), patch.object(agent, "emit"):
            self.assertEqual(agent._web_research_query(
                self.messages, "model", self.endpoint,
            ).query, "current officeholder")

    def test_stable_question_does_not_force_search(self):
        with patch.object(agent, "llm_text", return_value='{"needs_web_search":false}'):
            self.assertEqual(agent._web_research_query(self.messages, "model", self.endpoint).query, "")

    def test_local_task_does_not_call_web_classifier(self):
        with patch.object(agent, "llm_text") as invoke:
            self.assertFalse(agent._web_research_query(
                [HumanMessage(content="Read ./server.js")], "model", self.endpoint,
            ).query)
        invoke.assert_not_called()

    def test_unsupported_answer_is_revised_before_return(self):
        wrong = f"The current leader is Person A. Source: {self.url}"
        corrected = f"The current leader is Person B. Source: {self.url}"
        with patch.object(agent, "llm_text", side_effect=[
            wrong, '{"approved":false,"reason":"The page identifies Person B."}',
            corrected, '{"approved":true,"reason":"Supported by the page."}',
        ]) as invoke, patch.object(agent, "emit"):
            self.assertEqual(agent._synthesize_answer(
                "model", self.endpoint, self.evidence,
            ), corrected)
        initial_prompt = invoke.call_args_list[0].args[2][1].content
        self.assertIn("tool: fetch_web_page:", initial_prompt)
        self.assertIn("Original transliterated question", initial_prompt)
        self.assertNotIn("Translated question", initial_prompt)

    def test_uncited_answer_is_not_emitted(self):
        with patch.object(agent, "llm_text", return_value="An unsupported answer."), patch.object(agent, "emit"):
            with self.assertRaisesRegex(ValueError, "must cite a URL"):
                agent._synthesize_answer("model", self.endpoint, self.evidence)

    def test_repeatedly_unsupported_answer_fails_explicitly(self):
        with patch.object(agent, "llm_text", side_effect=[
            f"Person A. {self.url}", '{"approved":false,"reason":"Unsupported person."}',
            f"Person A. {self.url}", '{"approved":false,"reason":"Unsupported person."}',
        ]), patch.object(agent, "emit"):
            with self.assertRaisesRegex(ValueError, "Unable to verify"):
                agent._synthesize_answer("model", self.endpoint, self.evidence)

    def test_unfetched_citation_is_rejected(self):
        with patch.object(agent, "llm_text", return_value=(
            f"Person B. {self.url} and https://example.org/invented"
        )), patch.object(agent, "emit"):
            with self.assertRaisesRegex(ValueError, "URLs that were not fetched"):
                agent._synthesize_answer("model", self.endpoint, self.evidence)

    def test_citation_parser_preserves_balanced_url_delimiters(self):
        urls = [
            "https://example.org/wiki/Person_(politician)",
            "https://example.org/wiki/Person_(role_(title))",
            "https://example.org/list?item[0]=leader",
        ]
        for url in urls:
            for citation in (url, f"[source]({url}).", f"({url})", f"<{url}>", f"`{url}`"):
                with self.subTest(citation=citation):
                    self.assertEqual(agent._citation_urls(citation), {url})

    def test_parenthesized_fetched_citation_passes_grounding(self):
        url = "https://example.org/wiki/Person_(politician)"
        evidence = self.messages + [ToolMessage(
            content=f"URL: {url}\nPage content: The current leader is Person B.",
            name="fetch_web_page", tool_call_id="page-1",
        )]
        answer = f"Person B. [source]({url})."
        with patch.object(agent, "llm_text", side_effect=[
            answer, '{"approved":true,"reason":"Supported."}',
        ]) as invoke:
            self.assertEqual(agent._synthesize_answer("model", self.endpoint, evidence), answer)
        self.assertEqual(invoke.call_count, 2)

    def test_unfetched_url_with_allowed_prefix_is_not_accepted(self):
        with patch.object(agent, "llm_text", return_value=f"Person B. {self.url}/invented"), \
                patch.object(agent, "emit"):
            with self.assertRaisesRegex(ValueError, "URLs that were not fetched"):
                agent._synthesize_answer("model", self.endpoint, self.evidence)

    def test_unfetched_url_retry_includes_exact_allowed_urls(self):
        corrected = f"Person B. [source]({self.url})."
        with patch.object(agent, "llm_text", side_effect=[
            "Person B. https://example.org/invented",
            corrected, '{"approved":true,"reason":"Supported."}',
        ]) as invoke, patch.object(agent, "emit"):
            self.assertEqual(agent._synthesize_answer(
                "model", self.endpoint, self.evidence,
            ), corrected)
        retry_prompt = invoke.call_args_list[1].args[2][-1].content
        self.assertIn(f"Allowed citation URLs:\n{self.url}", retry_prompt)
        self.assertIn("https://example.org/invented", retry_prompt)

    def test_web_route_uses_normalized_query_and_verified_synthesis(self):
        with patch.object(agent, "_web_research_query", return_value=agent.ResearchDecision(query="normalized query")), \
                patch.object(agent, "_prepare_research", return_value=(self.evidence, [self.url])) as research, \
                patch.object(agent, "_synthesize_answer", return_value="Verified answer") as synthesize, \
                patch.object(agent, "_detect_kind") as detect, patch.object(agent, "emit") as emit:
            agent.run_code_assistant(self.messages, "model", self.endpoint)
        research.assert_called_once_with(self.messages, "normalized query")
        synthesize.assert_called_once_with("model", self.endpoint, self.evidence)
        detect.assert_not_called()
        self.assertIn(call({"message": {"content": "Verified answer"}}), emit.call_args_list)

    def test_web_failure_does_not_fall_back_to_memory(self):
        with patch.object(agent, "_web_research_query", return_value=agent.ResearchDecision(query="normalized query")), \
                patch.object(agent, "_prepare_research", side_effect=ValueError("No readable pages")), \
                patch.object(agent, "_synthesize_answer") as synthesize, patch.object(agent, "emit") as emit:
            with self.assertRaisesRegex(ValueError, "No readable pages"):
                agent.run_code_assistant(self.messages, "model", self.endpoint)
        synthesize.assert_not_called()
        self.assertFalse(any("done" in invocation.args[0] for invocation in emit.call_args_list))

    def test_clarification_returns_before_search_or_answer_in_both_modes(self):
        question = "Do you mean the state's Chief Minister or the country's Prime Minister?"
        for runner in (
            lambda: agent.run_code_assistant(self.messages, "model", self.endpoint),
            lambda: agent.run_tool_agent(None, self.messages, "model", self.endpoint),
        ):
            with self.subTest(runner=runner), \
                    patch.object(agent, "_web_research_query", return_value=agent.ResearchDecision(
                        clarification=question,
                    )), patch.object(agent, "_prepare_research") as research, \
                    patch.object(agent, "_synthesize_answer") as synthesize, \
                    patch.object(agent, "emit") as emit:
                runner()
            research.assert_not_called()
            synthesize.assert_not_called()
            self.assertIn(call({"message": {"content": question}}), emit.call_args_list)
            self.assertIn(call({"done": True}), emit.call_args_list)

    def test_semantic_decision_can_ask_clarification(self):
        question = "Which office did you mean?"
        with patch.object(agent, "_intent_clarification", return_value=question):
            decision = agent._web_research_query(self.messages, "model", self.endpoint)
        self.assertEqual(decision.clarification, question)
        self.assertEqual(decision.query, "")

    def test_intent_clarification_precedes_search_classification(self):
        with patch.object(agent, "_intent_clarification", return_value="Which office did you mean?"), \
                patch.object(agent, "llm_text") as invoke:
            decision = agent._web_research_query(self.messages, "model", self.endpoint)
        self.assertEqual(decision.clarification, "Which office did you mean?")
        invoke.assert_not_called()

    def test_follow_up_context_reaches_semantic_decision(self):
        messages = [
            HumanMessage(content="Who leads the state?"),
            agent.AIMessage(content="Which office did you mean?"),
            HumanMessage(content="The Chief Minister"),
        ]
        with patch.object(agent, "llm_text", return_value=(
            '{"needs_web_search":true,"search_query":"current state chief minister","clarification":""}'
        )) as invoke:
            agent._web_research_query(messages, "model", self.endpoint)
        self.assertIn("Which office did you mean?", invoke.call_args.args[2][1].content)


class IntentClarificationTests(unittest.TestCase):
    def test_clear_status_does_not_show_unnecessary_confirmation(self):
        for request in ("Explain recursion", "Write a short birthday greeting", "What is 2 + 2?"):
            with self.subTest(request=request), patch.object(agent, "llm_text", return_value=(
                '{"status":"clear","reason":"","clarification":"Please confirm your request?"}'
            )):
                self.assertEqual(agent._intent_clarification(
                    [HumanMessage(content=request)], "http://localhost",
                ), "")

    def test_essential_missing_information_returns_question(self):
        with patch.object(agent, "llm_text", return_value=(
            '{"status":"needs_clarification","reason":"No destination was supplied",'
            '"clarification":"Which destination are you travelling to?"}'
        )):
            self.assertEqual(agent._intent_clarification(
                [HumanMessage(content="Plan a route to my destination")], "http://localhost",
            ), "Which destination are you travelling to?")

    def test_original_request_is_separate_from_follow_up_context(self):
        messages = [
            HumanMessage(content="Earlier ambiguous question"),
            agent.AIMessage(content="Earlier alternatives to choose from"),
            HumanMessage(content="Processed question", additional_kwargs={
                "original_request": "Clear original question",
            }),
        ]
        with patch.object(agent, "llm_text", return_value=(
            '{"status":"clear","reason":"","clarification":""}'
        )) as invoke:
            agent._intent_clarification(messages, "http://localhost")
        prompt = invoke.call_args.args[2]
        self.assertEqual(len(prompt), 3)
        self.assertEqual(prompt[-1].content, "Clear original question")
        self.assertIn("Context only", prompt[1].content)
        self.assertIn("Earlier alternatives", prompt[1].content)
        self.assertIn("Suggested translation or rewrite (a hint only", prompt[1].content)
        self.assertIn("Processed question", prompt[1].content)
        self.assertNotIn("Processed question", prompt[-1].content)

    def test_missing_status_retries_instead_of_showing_question(self):
        with patch.object(agent, "llm_text", side_effect=[
            '{"clarification":"Unnecessary confirmation?"}',
            '{"status":"clear","reason":"","clarification":""}',
        ]) as invoke, patch.object(agent, "emit"):
            self.assertEqual(agent._intent_clarification(
                [HumanMessage(content="Clear question")], "http://localhost",
            ), "")
        self.assertEqual(invoke.call_count, 2)

    def test_unjustified_clarification_retries(self):
        with patch.object(agent, "llm_text", side_effect=[
            '{"status":"needs_clarification","reason":"","clarification":"Are you sure?"}',
            '{"status":"clear","reason":"","clarification":""}',
        ]) as invoke, patch.object(agent, "emit"):
            self.assertEqual(agent._intent_clarification(
                [HumanMessage(content="Write a short birthday greeting")], "http://localhost",
            ), "")
        self.assertEqual(invoke.call_count, 2)

    def test_local_task_can_request_clarification_before_any_tools(self):
        with patch.object(agent, "_intent_clarification", return_value="Which file should be deleted?"), \
                patch.object(agent, "llm_text") as invoke:
            decision = agent._web_research_query(
                [HumanMessage(content="Delete the file")], "model", "http://localhost",
            )
        self.assertEqual(decision.clarification, "Which file should be deleted?")
        self.assertEqual(decision.query, "")
        invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
