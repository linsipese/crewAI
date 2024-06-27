from copy import deepcopy
import os
import uuid
from typing import Any, Dict, List, Optional

from langchain.agents.agent import RunnableAgent
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import (
    UUID4,
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.agents import CacheHandler, CrewAgentExecutor, ToolsHandler
from crewai.agents.executor import MuiltAgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.utilities import I18N, Logger, RPMController
from crewai.utilities.token_counter_callback import TokenCalcHandler, TokenProcess

DEFAULT_REASONING_MODULES = [
    "1. How could I devise an experiment to help solve that problem?",
    "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    # "3. How could I measure progress on this problem?",
    "4. How can I simplify the problem so that it is easier to solve?",
    "5. What are the key assumptions underlying this problem?",
    "6. What are the potential risks and drawbacks of each solution?",
    "7. What are the alternative perspectives or viewpoints on this problem?",
    "8. What are the long-term implications of this problem and its solutions?",
    "9. How can I break down this problem into smaller, more manageable parts?",
    "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    # "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    # "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "16. What is the core issue or problem that needs to be addressed?",
    "17. What are the underlying causes or factors contributing to the problem?",
    "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "19. What are the potential obstacles or challenges that might arise in solving this problem?",
    "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "23. How can progress or success in solving the problem be measured or evaluated?",
    "24. What indicators or metrics can be used?",
    "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "30. Is the problem a design challenge that requires creative solutions and innovation?",
    "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "33. What kinds of solution typically are produced for this kind of problem specification?",
    "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
    "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
    "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
    "37. Ignoring the current best solution, create an entirely new solution to the problem."
    # "38. Let’s think step by step."
    "39. Let’s make a step by step plan and implement it with good notation and explanation.",
]


class SelfDiscoverAgent(Agent):
    """Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate tasks to other agents.

    Attributes:
            agent_executor: An instance of the CrewAgentExecutor class.
            role: The role of the agent.
            goal: The objective of the agent.
            backstory: The backstory of the agent.
            config: Dict representation of agent configuration.
            llm: The language model that will run the agent.
            function_calling_llm: The language model that will handle the tool calling for this agent, it overrides the crew function_calling_llm.
            max_iter: Maximum number of iterations for an agent to execute a task.
            memory: Whether the agent should have memory or not.
            max_rpm: Maximum number of requests per minute for the agent execution to be respected.
            verbose: Whether the agent execution should be in verbose mode.
            allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
            tools: Tools at agents disposal
            step_callback: Callback to be executed after each step of the agent execution.
            callbacks: A list of callback functions from the langchain library that are triggered during the agent's execution process
    """

    __hash__ = object.__hash__  # type: ignore
    _logger: Logger = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr(default=None)
    _request_within_rpm_limit: Any = PrivateAttr(default=None)
    _token_process: TokenProcess = TokenProcess()

    formatting_errors: int = 0
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    cache: bool = Field(
        default=True,
        description="Whether the agent should use a cache for tool usage.",
    )
    config: Optional[Dict[str, Any]] = Field(
        description="Configuration for the agent",
        default=None,
    )
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the agent execution to be respected.",
    )
    verbose: bool = Field(
        default=False, description="Verbose mode for the Agent Execution"
    )
    allow_delegation: bool = Field(
        default=True, description="Allow delegation of tasks to agents"
    )
    tools: Optional[List[Any]] = Field(
        default_factory=list, description="Tools at agents disposal"
    )
    max_iter: Optional[int] = Field(
        default=25, description="Maximum iterations for an agent to execute a task"
    )
    max_execution_time: Optional[int] = Field(
        default=None,
        description="Maximum execution time for an agent to execute a task",
    )
    select_executor: InstanceOf[CrewAgentExecutor] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    adapt_executor: InstanceOf[CrewAgentExecutor] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    structured_executor: InstanceOf[CrewAgentExecutor] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    reasoning_executor: InstanceOf[CrewAgentExecutor] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    reasoning_modules: List[str] = Field(
        default=DEFAULT_REASONING_MODULES,
        description="Modules to be used in the reasoning process.",
    )
    crew: Any = Field(default=None, description="Crew to which the agent belongs.")
    tools_handler: InstanceOf[ToolsHandler] = Field(
        default=None, description="An instance of the ToolsHandler class."
    )
    cache_handler: InstanceOf[CacheHandler] = Field(
        default=None, description="An instance of the CacheHandler class."
    )
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")
    llm: Any = Field(
        default_factory=lambda: ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
        ),
        description="Language model that will run the agent.",
    )
    function_calling_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    callbacks: Optional[List[InstanceOf[BaseCallbackHandler]]] = Field(
        default=None, description="Callback to be executed"
    )

    _original_role: str | None = None
    _original_goal: str | None = None
    _original_backstory: str | None = None

    def __init__(__pydantic_self__, **data):
        config = data.pop("config", {})
        super().__init__(**config, **data)

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def set_attributes_based_on_config(self) -> "Agent":
        """Set attributes based on the agent configuration."""
        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self

    @model_validator(mode="after")
    def set_private_attrs(self):
        """Set private attributes."""
        self._logger = Logger(self.verbose)
        if self.max_rpm and not self._rpm_controller:
            self._rpm_controller = RPMController(
                max_rpm=self.max_rpm, logger=self._logger
            )
        return self

    @model_validator(mode="after")
    def set_agent_executor(self) -> "Agent":
        """set agent executor is set."""
        if hasattr(self.llm, "model_name"):
            token_handler = TokenCalcHandler(self.llm.model_name, self._token_process)

            # Ensure self.llm.callbacks is a list
            if not isinstance(self.llm.callbacks, list):
                self.llm.callbacks = []

            # Check if an instance of TokenCalcHandler already exists in the list
            if not any(
                isinstance(handler, TokenCalcHandler) for handler in self.llm.callbacks
            ):
                self.llm.callbacks.append(token_handler)

        if not self.agent_executor:
            if not self.cache_handler:
                self.cache_handler = CacheHandler()
            self.set_cache_handler(self.cache_handler)
        return self

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """
        if self.tools_handler:
            # type: ignore # Incompatible types in assignment (expression has type "dict[Never, Never]", variable has type "ToolCalling")
            self.tools_handler.last_used_tool = {}

        task_prompt = task.prompt()

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self.crew and self.crew.memory:
            contextual_memory = ContextualMemory(
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
            )
            memory = contextual_memory.build_context_for_task(task, context)
            if memory.strip() != "":
                task_prompt += self.i18n.slice("memory").format(memory=memory)

        self.create_agent_executor()
        self.select_executor.task = task
        select_result = self.select_executor.invoke(
            {
                "task_description": task_prompt,
                "reasoning_modules": "\n".join(self.reasoning_modules),
            }
        )["output"]
        self.adapt_executor.task = task
        adapt_result = self.adapt_executor.invoke(
            {
                "selected_modules": select_result,
                "task_description": task_prompt,
            }
        )["output"]
        self.structured_executor.task = task
        structured_result = self.structured_executor.invoke(
            {
                "adapted_modules": adapt_result,
                "task_description": task_prompt,
            }
        )["output"]
        self.reasoning_executor.task = task
        result = self.reasoning_executor.invoke(
            {
                "reasoning_structure": structured_result,
                "task_description": task_prompt,
            }
        )["output"]

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return result

    def set_cache_handler(self, cache_handler: CacheHandler) -> None:
        """Set the cache handler for the agent.

        Args:
            cache_handler: An instance of the CacheHandler class.
        """
        self.tools_handler = ToolsHandler()
        if self.cache:
            self.cache_handler = cache_handler
            self.tools_handler.cache = cache_handler
        self.create_agent_executor()

    def set_rpm_controller(self, rpm_controller: RPMController) -> None:
        """Set the rpm controller for the agent.

        Args:
            rpm_controller: An instance of the RPMController class.
        """
        if not self._rpm_controller:
            self._rpm_controller = rpm_controller
            self.create_agent_executor()

    def create_agent_executor(self, tools=None) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        tools = tools or self.tools

        select_args = {
            "task_description": lambda x: x["task_description"],
            "reasoning_modules": lambda x: x["reasoning_modules"],
        }
        adapt_args = {
            "task_description": lambda x: x["task_description"],
            "selected_modules": lambda x: x["selected_modules"],
        }
        structured_args = {
            "task_description": lambda x: x["task_description"],
            "adapted_modules": lambda x: x["adapted_modules"],
        }
        reasoning_args = {
            "task_description": lambda x: x["task_description"],
            "reasoning_structure": lambda x: x["reasoning_structure"],
        }

        executor_args = {
            "llm": self.llm,
            "i18n": self.i18n,
            "crew": self.crew,
            "crew_agent": self,
            "tools": self._parse_tools(tools),
            "verbose": self.verbose,
            "original_tools": tools,
            "handle_parsing_errors": True,
            "max_iterations": self.max_iter,
            "max_execution_time": self.max_execution_time,
            "step_callback": self.step_callback,
            "tools_handler": self.tools_handler,
            "function_calling_llm": self.function_calling_llm,
            "callbacks": self.callbacks,
        }

        if self._rpm_controller:
            executor_args["request_within_rpm_limit"] = (
                self._rpm_controller.check_or_wait
            )

        select_prompt = PromptTemplate.from_template(
            template=self.i18n.slice("select_prompt"),
        )
        adapt_prompt = PromptTemplate.from_template(
            template=self.i18n.slice("adapt_prompt"),
        )
        structured_prompt = PromptTemplate.from_template(
            template=self.i18n.slice("structured_prompt"),
        )
        reasoning_prompt = PromptTemplate.from_template(
            template=self.i18n.slice("reasoning_prompt"),
        )

        bind = self.llm.bind()
        select_chain = select_args | select_prompt | bind | StrOutputParser()
        self.select_executor = MuiltAgentExecutor(
            agent=RunnableAgent(runnable=select_chain), **executor_args
        )
        adapt_chain = adapt_args | adapt_prompt | bind | StrOutputParser()
        self.adapt_executor = MuiltAgentExecutor(
            agent=RunnableAgent(runnable=adapt_chain), **executor_args
        )
        structured_chain = (
            structured_args | structured_prompt | bind | StrOutputParser()
        )
        self.structured_executor = MuiltAgentExecutor(
            agent=RunnableAgent(runnable=structured_chain), **executor_args
        )
        reasoning_chain = reasoning_args | reasoning_prompt | bind | StrOutputParser()
        self.reasoning_executor = MuiltAgentExecutor(
            agent=RunnableAgent(runnable=reasoning_chain), **executor_args
        )

    def copy(self):
        """Create a deep copy of the Agent."""
        exclude = {
            "id",
            "_logger",
            "_rpm_controller",
            "_request_within_rpm_limit",
            "_token_process",
            "agent_executor",
            "tools",
            "tools_handler",
            "cache_handler",
        }

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}

        copied_agent = SelfDiscoverAgent(**copied_data)
        copied_agent.tools = deepcopy(self.tools)

        return copied_agent


class ReflectionAgent(Agent):
    """Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate tasks to other agents.

    Attributes:
            agent_executor: An instance of the CrewAgentExecutor class.
            role: The role of the agent.
            goal: The objective of the agent.
            backstory: The backstory of the agent.
            config: Dict representation of agent configuration.
            llm: The language model that will run the agent.
            function_calling_llm: The language model that will handle the tool calling for this agent, it overrides the crew function_calling_llm.
            max_iter: Maximum number of iterations for an agent to execute a task.
            memory: Whether the agent should have memory or not.
            max_rpm: Maximum number of requests per minute for the agent execution to be respected.
            verbose: Whether the agent execution should be in verbose mode.
            allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
            tools: Tools at agents disposal
            step_callback: Callback to be executed after each step of the agent execution.
            callbacks: A list of callback functions from the langchain library that are triggered during the agent's execution process
    """

    __hash__ = object.__hash__  # type: ignore
    _logger: Logger = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr(default=None)
    _request_within_rpm_limit: Any = PrivateAttr(default=None)
    _token_process: TokenProcess = TokenProcess()

    formatting_errors: int = 0
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    cache: bool = Field(
        default=True,
        description="Whether the agent should use a cache for tool usage.",
    )
    config: Optional[Dict[str, Any]] = Field(
        description="Configuration for the agent",
        default=None,
    )
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the agent execution to be respected.",
    )
    verbose: bool = Field(
        default=False, description="Verbose mode for the Agent Execution"
    )
    allow_delegation: bool = Field(
        default=True, description="Allow delegation of tasks to agents"
    )
    tools: Optional[List[Any]] = Field(
        default_factory=list, description="Tools at agents disposal"
    )
    max_iter: Optional[int] = Field(
        default=25, description="Maximum iterations for an agent to execute a task"
    )
    max_execution_time: Optional[int] = Field(
        default=None,
        description="Maximum execution time for an agent to execute a task",
    )
    generate_system: str = Field(
        default="You are an essay assistant tasked with writing excellent 5-paragraph essays.\n Generate the best essay possible for the user's request.\n If the user provides critique, respond with a revised version of your previous attempts.",
        description="The system to generate the output.",
    )
    reflect_system: str = Field(
        default="You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission.\n Provide detailed recommendations, including requests for length, depth, style, etc.",
        description="The system to reflect the output.",
    )
    generate_executor: InstanceOf[CrewAgentExecutor] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    reflect_executor: InstanceOf[CrewAgentExecutor] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    crew: Any = Field(default=None, description="Crew to which the agent belongs.")
    tools_handler: InstanceOf[ToolsHandler] = Field(
        default=None, description="An instance of the ToolsHandler class."
    )
    cache_handler: InstanceOf[CacheHandler] = Field(
        default=None, description="An instance of the CacheHandler class."
    )
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")
    llm: Any = Field(
        default_factory=lambda: ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
        ),
        description="Language model that will run the agent.",
    )
    function_calling_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    callbacks: Optional[List[InstanceOf[BaseCallbackHandler]]] = Field(
        default=None, description="Callback to be executed"
    )

    _original_role: str | None = None
    _original_goal: str | None = None
    _original_backstory: str | None = None

    def __init__(__pydantic_self__, **data):
        config = data.pop("config", {})
        super().__init__(**config, **data)

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def set_attributes_based_on_config(self) -> "Agent":
        """Set attributes based on the agent configuration."""
        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self

    @model_validator(mode="after")
    def set_private_attrs(self):
        """Set private attributes."""
        self._logger = Logger(self.verbose)
        if self.max_rpm and not self._rpm_controller:
            self._rpm_controller = RPMController(
                max_rpm=self.max_rpm, logger=self._logger
            )
        return self

    @model_validator(mode="after")
    def set_agent_executor(self) -> "Agent":
        """set agent executor is set."""
        if hasattr(self.llm, "model_name"):
            token_handler = TokenCalcHandler(self.llm.model_name, self._token_process)

            # Ensure self.llm.callbacks is a list
            if not isinstance(self.llm.callbacks, list):
                self.llm.callbacks = []

            # Check if an instance of TokenCalcHandler already exists in the list
            if not any(
                isinstance(handler, TokenCalcHandler) for handler in self.llm.callbacks
            ):
                self.llm.callbacks.append(token_handler)

        if not self.agent_executor:
            if not self.cache_handler:
                self.cache_handler = CacheHandler()
            self.set_cache_handler(self.cache_handler)
        return self

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """
        if self.tools_handler:
            # type: ignore # Incompatible types in assignment (expression has type "dict[Never, Never]", variable has type "ToolCalling")
            self.tools_handler.last_used_tool = {}

        task_prompt = task.prompt()

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self.crew and self.crew.memory:
            contextual_memory = ContextualMemory(
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
            )
            memory = contextual_memory.build_context_for_task(task, context)
            if memory.strip() != "":
                task_prompt += self.i18n.slice("memory").format(memory=memory)

        self.create_agent_executor()
        self.generate_executor.task = task
        self.reflect_executor.task = task
        request = HumanMessage(
            content="Write an essay on why the little prince is relevant in modern childhood"
        )
        messages = [request]
        for _ in range(self.max_iter-1):
            generate_result = self.generate_executor.invoke(
                {
                    "massages": messages,
                }
            )["output"]
            messages.append(AIMessage(content=generate_result))
            reflect_result = self.reflect_executor.invoke(
                {
                    "massages": messages,
                }
            )["output"]
            messages.append(HumanMessage(content=reflect_result))
        result = self.generate_executor.invoke(
            {
                "massages": messages,
            }
        )["output"]
        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return result

    def set_cache_handler(self, cache_handler: CacheHandler) -> None:
        """Set the cache handler for the agent.

        Args:
            cache_handler: An instance of the CacheHandler class.
        """
        self.tools_handler = ToolsHandler()
        if self.cache:
            self.cache_handler = cache_handler
            self.tools_handler.cache = cache_handler
        self.create_agent_executor()

    def set_rpm_controller(self, rpm_controller: RPMController) -> None:
        """Set the rpm controller for the agent.

        Args:
            rpm_controller: An instance of the RPMController class.
        """
        if not self._rpm_controller:
            self._rpm_controller = rpm_controller
            self.create_agent_executor()

    def create_agent_executor(self, tools=None) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        tools = tools or self.tools

        generate_args = {
            "messages": lambda x: x["messages"],
        }
        cls_map = {"ai": HumanMessage, "human": AIMessage ,"system": SystemMessage}
        reflection_args = {
            "messages": lambda x: [cls_map[i.type](content=i.content) for i in x["messages"]],
        }

        executor_args = {
            "llm": self.llm,
            "i18n": self.i18n,
            "crew": self.crew,
            "crew_agent": self,
            "tools": self._parse_tools(tools),
            "verbose": self.verbose,
            "original_tools": tools,
            "handle_parsing_errors": True,
            "max_iterations": self.max_iter,
            "max_execution_time": self.max_execution_time,
            "step_callback": self.step_callback,
            "tools_handler": self.tools_handler,
            "function_calling_llm": self.function_calling_llm,
            "callbacks": self.callbacks,
        }

        if self._rpm_controller:
            executor_args["request_within_rpm_limit"] = (
                self._rpm_controller.check_or_wait
            )
        generate_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.generate_system,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.reflect_system,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )


        bind = self.llm.bind()
        generate_chain = generate_args | generate_prompt | bind | StrOutputParser()
        self.generate_executor = MuiltAgentExecutor(
            agent=RunnableAgent(runnable=generate_chain), **executor_args
        )
        reflection_chain = reflection_args | reflection_prompt | bind | StrOutputParser()
        self.reflection_executor = MuiltAgentExecutor(
            agent=RunnableAgent(runnable=reflection_chain), **executor_args
        )


    def copy(self):
        """Create a deep copy of the Agent."""
        exclude = {
            "id",
            "_logger",
            "_rpm_controller",
            "_request_within_rpm_limit",
            "_token_process",
            "agent_executor",
            "tools",
            "tools_handler",
            "cache_handler",
        }

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}

        copied_agent = ReflectionAgent(**copied_data)
        copied_agent.tools = deepcopy(self.tools)

        return copied_agent
