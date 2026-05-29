import operator
from datetime import datetime
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from passive_goal_creator.main import Goal, PassiveGoalCreator
from prompt_optimizer.main import OptimizedGoal, PromptOptimizer
from pydantic import BaseModel, Field
from response_optimizer.main import ResponseOptimizer

class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="3~5개로 분해된 태스크"
    )

class SinglePathPlanGenerationState(BaseModel):
    query: str = Field(..., description="사용자가 입력한 쿼리")
    optimized_goal: str = Field(default="", description="최적화된 목표")
    optimized_response: str = Field(default="", description="최적화된 응답 정의")

    tasks: list[str] = Field(default_factory=list, description="실행할 태스크 리스트")
    current_task_index : int = Field(default=0, description="현재 실행중인 태스크 번호")
    results: Annotated[list[str], operator.add] = Field(default_factory=list, description="실행 완료된 태스크 결과 리스트")

    final_output: str = Field(default="", description="최종 출력 결과")


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_date}\n"
            "-----\n"
            "태스크: 주어진 목표를 구체적이고 실행 가능한 태스크로 분해해 주세요.\n"
            "요건:\n"
            "1. 다음 행동만으로 목표를 달성할 것. 절대 지정된 이외의 행동을 취하지 말 것.\n"
            "   - 인터넷을 이용하여 목표 달성을 위한 조사를 수행한다.\n"
            "2. 각 태스크는 구체적이고 상세하게 기재하며, 단독으로 실행 및 검증 가능한 정보를 포함할 것. 추상적인 표현을 일절 포함하지 말 것.\n"
            "3. 태스크는 실행 가능한 순서로 리스트화할 것.\n"
            "4. 태스크는 한국어로 출력할 것.\n"
            "목표: {query}"
        )

        chain = prompt | self.llm.with_structured_output(DecomposedTasks)

        return chain.invoke({"query" : query})

class TaskExecutor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]

