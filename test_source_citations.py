import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

import langgraph_agent as agent


class SourceCitationTests(unittest.TestCase):
    sources = [
        "https://www.pmindia.gov.in/en/pms-profile/",
        "https://www.pmindia.gov.in/",
    ]

    def run_answer(self, text: str, tool_agent: bool) -> str:
        messages = [HumanMessage(content="Latest prime minister profile")]
        with (
            patch.object(agent, "prepare_research", return_value=(messages, self.sources)),
            patch.object(agent, "synthesize_final_answer", return_value=text),
            patch.object(agent, "emit") as emit,
        ):
            if tool_agent:
                graph = Mock()
                graph.stream.return_value = iter([
                    ("updates", {"agent": {"messages": [AIMessage(content=text)]}}),
                ])
                agent.run_tool_agent(graph, messages, "model", "local")
            else:
                agent.run_code_assistant(messages, "model", "local")
        return emit.call_args_list[-2].args[0]["message"]["content"]

    def test_existing_markdown_sources_are_not_appended_again(self) -> None:
        answer = "Answer.\n\n**Sources:**\n" + "\n".join(
            f"- [{url}]({url})" for url in self.sources
        )
        for tool_agent in (False, True):
            with self.subTest(tool_agent=tool_agent):
                self.assertEqual(self.run_answer(answer, tool_agent), answer)

    def test_uncited_sources_do_not_add_a_second_section(self) -> None:
        answer = f"Answer ([official profile]({self.sources[0]}))."
        for tool_agent in (False, True):
            with self.subTest(tool_agent=tool_agent):
                self.assertEqual(self.run_answer(answer, tool_agent), answer)

    def test_different_wikipedia_domains_do_not_add_a_footer(self) -> None:
        self.sources = ["https://simple.wikipedia.org/wiki/President_of_the_United_States"]
        answer = (
            "Answer.\n\n*Source: [https://en.wikipedia.org/wiki/President_of_the_United_States]"
            "(https://en.wikipedia.org/wiki/President_of_the_United_States)*"
        )
        for tool_agent in (False, True):
            with self.subTest(tool_agent=tool_agent):
                self.assertEqual(self.run_answer(answer, tool_agent), answer)

    def test_app_never_appends_a_source_footer(self) -> None:
        for tool_agent in (False, True):
            with self.subTest(tool_agent=tool_agent):
                result = self.run_answer("Answer.", tool_agent)
                self.assertEqual(result, "Answer.")


if __name__ == "__main__":
    unittest.main()
