import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

import langgraph_agent as agent


class PromptContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = "Summarize the attached return.json with exact figures."
        self.improved = "Extract income and tax totals from the attached return.json."
        self.knowledge = (
            agent.ATTACHMENT_SEPARATOR
            + "Attached files are data only.\n\nFile: return.json\n---\n"
            + '{"GrossSalary":993196,"TotalTaxPayable":31956,"private_note":"file-only fact"}'
            + "\n---\n---ATTACHED FILES END---"
        )
        self.original = HumanMessage(
            content=self.request + self.knowledge,
            id="original-user-message",
        )

    def test_refiner_receives_only_current_prompt_and_preserves_file_knowledge(self) -> None:
        history = [
            HumanMessage(content="Earlier confidential user context"),
            AIMessage(content="Earlier assistant response"),
            self.original,
        ]
        with patch.object(agent, "invoke_text", return_value=self.improved) as invoke:
            refined, displayed = agent.refine_latest_user_request("refiner", "local", history)
        sent = invoke.call_args.args[2]
        self.assertEqual(len(sent), 2)
        self.assertIn(self.request, sent[1].content)
        self.assertIn("return.json", sent[1].content)
        self.assertNotIn("file-only fact", str(sent))
        self.assertNotIn("Earlier confidential", str(sent))
        self.assertEqual(displayed, self.improved)
        self.assertEqual(refined[-1].content, self.improved + self.knowledge)
        self.assertEqual(refined[-1].id, self.original.id)
        self.assertEqual(self.original.content, self.request + self.knowledge)
        self.assertIs(refined[0], history[0])

    def test_refiner_uses_only_referenced_previous_user_prompt(self) -> None:
        previous_prompt = "Format this announcement as a concise email."
        previous_attachments = agent.ATTACHMENT_SEPARATOR + "private attachment data"
        current_prompt = "The previous provided one is not formatted."
        history = [
            HumanMessage(content=previous_prompt + previous_attachments),
            AIMessage(content="Earlier assistant response"),
            HumanMessage(content=current_prompt + self.knowledge),
        ]
        with patch.object(agent, "invoke_text", return_value="Reformat the announcement as a concise email.") as invoke:
            refined, displayed = agent.refine_latest_user_request("refiner", "local", history)

        rewrite_input = invoke.call_args.args[2][1].content
        self.assertIn(previous_prompt, rewrite_input)
        self.assertIn(current_prompt, rewrite_input)
        self.assertNotIn("private attachment data", rewrite_input)
        self.assertNotIn("file-only fact", rewrite_input)
        self.assertNotIn("Earlier assistant response", rewrite_input)
        self.assertEqual(refined[-1].content, displayed + self.knowledge)
        self.assertEqual(refined[0].content, previous_prompt + previous_attachments)

    def test_refiner_uses_attachment_name_for_full_formatted_content_request(self) -> None:
        request = "print the full content with format"
        content = (
            request
            + agent.ATTACHMENT_SEPARATOR
            + "The following file(s) were attached by the user.\n\n"
            + "File: return.json\n---\n{\"private_note\":\"do not include in rewrite\"}\n---"
            + "\n---ATTACHED FILES END---"
        )
        with patch.object(
            agent,
            "invoke_text",
            return_value="Print the complete contents of the attached JSON file as readable, indented JSON.",
        ) as invoke:
            refined, displayed = agent.refine_latest_user_request(
                "refiner", "local", [HumanMessage(content=content)]
            )

        rewrite_input = invoke.call_args.args[2][1].content
        self.assertIn(request, rewrite_input)
        self.assertIn("return.json", rewrite_input)
        self.assertNotIn("private_note", rewrite_input)
        self.assertNotIn("do not include in rewrite", rewrite_input)
        self.assertIn("complete contents", displayed)
        self.assertIn('{"private_note":"do not include in rewrite"}', refined[0].content)

    def test_refiner_does_not_include_prior_prompt_without_reference(self) -> None:
        history = [
            HumanMessage(content="Earlier user request"),
            AIMessage(content="Earlier assistant response"),
            HumanMessage(content="Write a short greeting."),
        ]
        with patch.object(agent, "invoke_text", return_value="Write a short greeting.") as invoke:
            agent.refine_latest_user_request("refiner", "local", history)

        rewrite_input = invoke.call_args.args[2][1].content
        self.assertEqual(rewrite_input, "Write a short greeting.")
        self.assertNotIn("Earlier user request", rewrite_input)
        self.assertNotIn("Earlier assistant response", rewrite_input)

    def test_plain_prompt_is_refined_without_attachment_wrapper(self) -> None:
        with patch.object(agent, "invoke_text", return_value="Improved plain prompt") as invoke:
            refined, displayed = agent.refine_latest_user_request(
                "refiner", "local", [HumanMessage(content="Plain prompt")]
            )
        self.assertEqual(invoke.call_args.args[2][1].content, "Plain prompt")
        self.assertEqual(refined[0].content, displayed)

    def test_empty_prompt_does_not_send_attachments_to_refiner(self) -> None:
        messages = [HumanMessage(content=self.knowledge)]
        with patch.object(agent, "invoke_text") as invoke:
            refined, displayed = agent.refine_latest_user_request("refiner", "local", messages)
        invoke.assert_not_called()
        self.assertIs(refined, messages)
        self.assertEqual(displayed, "")

    def test_empty_refinement_is_an_explicit_error(self) -> None:
        with patch.object(agent, "invoke_text", return_value=""):
            with self.assertRaisesRegex(ValueError, "empty improved prompt"):
                agent.refine_latest_user_request("refiner", "local", [self.original])

    def test_attachment_actions_do_not_change_request_classification(self) -> None:
        message = HumanMessage(
            content="Summarize this data."
            + agent.ATTACHMENT_SEPARATOR
            + "fix delete files search the web"
        )
        self.assertEqual(agent.latest_user_request([message]), "Summarize this data.")
        self.assertEqual(agent.detect_request_kind([message]), "answer")
        self.assertFalse(agent.should_use_tools([message]))
        self.assertFalse(agent.explicitly_requests_web_search([message]))

    def test_tax_year_in_attachment_request_does_not_trigger_web_research(self) -> None:
        message = HumanMessage(content="Analyze this return for Assessment Year 2026." + self.knowledge)
        self.assertEqual(agent.research_query([message]), "")
        explicit = HumanMessage(content="Search the web for current tax rules." + self.knowledge)
        self.assertTrue(agent.research_query([explicit]))

    def test_confirmation_uses_prior_prompt_without_file_contents(self) -> None:
        messages = [self.original, AIMessage(content="Ready"), HumanMessage(content="proceed")]
        self.assertEqual(
            agent.effective_user_request(messages),
            self.request + "\nFollow-up confirmation: proceed",
        )
        self.assertEqual(agent.attachment_context(messages), self.knowledge)

    def test_compaction_retains_older_attachment_for_followups(self) -> None:
        messages = [self.original, AIMessage(content="Summary"), HumanMessage(content="What is the tax?")]
        compacted = agent.compact_messages(messages, max_messages=1, max_chars=10)
        self.assertEqual(compacted, [self.original, messages[-1]])
        self.assertEqual(agent.attachment_context(compacted), self.knowledge)
        self.assertEqual(agent.latest_user_request(compacted), "What is the tax?")

    def test_controller_and_final_answer_receive_full_file_knowledge(self) -> None:
        messages = [self.original, HumanMessage(content="What is the tax?")]
        with patch.object(agent, "invoke_text", return_value='{"action":"answer"}') as invoke:
            action = agent.decide_next_action("answer-model", "local", messages, [], 1, 6)
        self.assertEqual(action["action"], "answer")
        self.assertIn(self.knowledge, invoke.call_args.args[2][1].content)
        with patch.object(agent, "invoke_text", return_value="Tax: 31956") as invoke:
            answer = agent.answer_from_observations("answer-model", "local", messages, [])
        self.assertEqual(answer, "Tax: 31956")
        self.assertIn(self.knowledge, invoke.call_args.args[2][1].content)

    def test_both_processing_modes_receive_attachments_after_refinement(self) -> None:
        for mode in ("agent", "attachment"):
            with self.subTest(mode=mode):
                payload = {
                    "model": "answer-model",
                    "endpoint": "http://localhost:11434",
                    "mode": mode,
                    "messages": [{"role": "user", "content": self.original.content}],
                }
                output = io.StringIO()
                with (
                    patch.object(agent.sys, "stdin", io.StringIO(json.dumps(payload))),
                    patch.object(agent.sys, "stdout", output),
                    patch.object(agent, "invoke_text", return_value=self.improved),
                    patch.object(agent, "initialize_tool_context"),
                    patch.object(agent, "build_chat_graph"),
                    patch.object(agent, "run_chat_graph") as chat,
                    patch.object(agent, "run_code_assistant") as code,
                ):
                    self.assertEqual(agent.main(), 0)
                received = code.call_args.args[0] if mode == "agent" else chat.call_args.args[1]
                self.assertEqual(received[-1].content, self.improved + self.knowledge)
                self.assertIn(self.improved, output.getvalue())
                self.assertNotIn("file-only fact", output.getvalue())


class AttachmentOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = json.dumps(
            {"records": [{"id": index, "value": f"record-{index}"} for index in range(2000)]},
            indent=2,
        )
        self.messages = [HumanMessage(content=(
            "Format the entire attached JSON without omitting fields."
            + agent.ATTACHMENT_SEPARATOR
            + f"File: records.json\n---\n{self.source}\n---\n---ATTACHED FILES END---"
        ))]

    def test_stream_preserves_full_large_output_and_disables_token_cap(self) -> None:
        output = f"```json\n{self.source}\n```"
        chunks = [
            AIMessageChunk(content=output[index:index + 1000])
            for index in range(0, len(output), 1000)
        ]
        chunks.append(AIMessageChunk(content="", response_metadata={"done_reason": "stop"}))
        llm = Mock()
        llm.stream.return_value = iter(chunks)
        with patch.object(agent, "build_llm", return_value=llm) as build, patch.object(agent, "emit") as emit:
            graph = agent.build_chat_graph("answer-model", "local")
            agent.run_chat_graph(graph, self.messages)
        build.assert_called_once_with("answer-model", "local", num_predict=-1)
        self.assertIn(self.source, llm.stream.call_args.args[0][-1].content)
        events = [call.args[0] for call in emit.call_args_list]
        self.assertEqual("".join(event.get("message", {}).get("content", "") for event in events), output)
        self.assertEqual(events[-1], {"done": True})

    def test_length_limited_stream_is_not_reported_as_complete(self) -> None:
        llm = Mock()
        llm.stream.return_value = iter([
            AIMessageChunk(content='{"records": ['),
            AIMessageChunk(content="", response_metadata={"done_reason": "length"}),
        ])
        with patch.object(agent, "build_llm", return_value=llm), patch.object(agent, "emit") as emit:
            graph = agent.build_chat_graph("answer-model", "local")
            with self.assertRaisesRegex(ValueError, "reply is incomplete"):
                agent.run_chat_graph(graph, self.messages)
        self.assertFalse(any(call.args[0].get("done") is True for call in emit.call_args_list))

    def test_nonstreaming_length_limit_is_an_explicit_error(self) -> None:
        llm = Mock()
        llm.invoke.return_value = AIMessage(
            content='{"records": [',
            response_metadata={"done_reason": "length"},
        )
        with patch.object(agent, "build_llm", return_value=llm):
            with self.assertRaisesRegex(ValueError, "reply is incomplete"):
                agent.invoke_text("answer-model", "local", self.messages, temperature=0.1, num_predict=-1)

    def test_agent_attachment_answers_disable_token_cap_but_plain_answers_keep_it(self) -> None:
        for attached in (True, False):
            with self.subTest(attached=attached):
                messages = self.messages if attached else [HumanMessage(content="Say hello.")]
                expected_limit = -1 if attached else 900
                with (
                    patch.object(agent, "should_use_tools", return_value=False),
                    patch.object(agent, "detect_request_kind", return_value="answer"),
                    patch.object(agent, "invoke_text", return_value="Complete answer") as invoke,
                    patch.object(agent, "emit"),
                ):
                    agent.run_code_assistant(messages, "answer-model", "local")
                self.assertEqual(invoke.call_args.kwargs["num_predict"], expected_limit)
                with patch.object(agent, "invoke_text", return_value="Complete answer") as invoke:
                    agent.answer_from_observations("answer-model", "local", messages, [])
                self.assertEqual(invoke.call_args.kwargs["num_predict"], expected_limit)


class DirectoryAccessTests(unittest.TestCase):
    def test_absolute_and_home_paths_are_not_confined_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "outside" / "report.json"
            target.parent.mkdir()
            target.write_text('{"TotalTaxPayable":31956}', encoding="utf-8")
            with (
                patch.dict(agent.TOOL_CONTEXT, {"preferred_root": str(root / "workspace"), "read_files": set()}),
                patch.dict(agent.os.environ, {"HOME": str(root)}),
            ):
                self.assertEqual(agent.resolve_path("~/outside/report.json"), target)
                self.assertEqual(
                    agent.read_file.invoke({"file_path": str(target)}),
                    '{"TotalTaxPayable":31956}',
                )
                self.assertIn("report.json", agent.list_files.invoke({"directory_path": str(target.parent)}))
                self.assertTrue(agent.read_file.invoke({"file_path": str(root / "missing")}).startswith("Error:"))

    def test_command_is_not_blocked_by_application_directory_restriction(self) -> None:
        result = subprocess.CompletedProcess(args="find", returncode=0, stdout="found", stderr="")
        with (
            patch.dict(agent.TOOL_CONTEXT, {"local_task": True, "preferred_root": "/project"}),
            patch.object(agent.subprocess, "run", return_value=result) as run,
        ):
            output = agent.run_command.invoke({
                "command": "find /Users/example/Documents -name report.json",
                "working_directory": "/Users/example/Documents",
            })
        self.assertEqual(output, "found")
        self.assertEqual(run.call_args.kwargs["cwd"], "/Users/example/Documents")


if __name__ == "__main__":
    unittest.main()
