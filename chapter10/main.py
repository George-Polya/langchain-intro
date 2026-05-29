import os
from dotenv import load_dotenv
import operator
from typing import Annotated, Any, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

LANGSMITH_TRACING=os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT=os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT=os.getenv("LANGSMITH_PROJECT")

# 페르소나를 나타내는 데이터 모델
class Persona(BaseModel):
    name: str = Field(..., description="페르소나의 이름")
    background: str = Field(..., description="페르소나의 배경")

# 페르소나 목록을 나타내는 데이터 모델
class Personas(BaseModel):
    personas: list[Persona] = Field(default_factory=list, description="페르소나 목록")

# 인터뷰 내용을 나타내는 데이터 모델
class Interview(BaseModel):
    persona: Persona = Field(..., description="인터뷰 대상 페르소나")
    question: str = Field(..., description="인터뷰 질문")
    answer: str = Field(... , description="인터뷰 응답")

# 인터뷰 결과 목록을 나타내는 데이터 모델
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(default_factory=list, description="인터뷰 결과목록")

# 평가 결과를 나타내는 데이터 모델
class EvaluationResult(BaseModel):
    reason: str = Field(..., description="판단 이유")
    is_sufficient: bool = Field(..., description="정보가 중요한지 여부")

# 요구사항 정의서 생성 AI 에이전트의 스테이트
class InterviewState(BaseModel):
    user_request: str = Field(..., description="사용자 요청")
    personas: Annotated[list[Persona], operator.add] = Field(default_factory=list, description="생성된 페르소나 목록")
    interviews: Annotated[list[Interview], operator.add] = Field(default_factory=list, description="실시된 인터뷰 목록")
    iteration: int = Field(default=0, description="페르소나 생성과 인터뷰의 반복 횟수")
    is_information_sufficient: bool = Field(default=False, description="정보가 충분한지 여부")
    
    requirements_doc: str = Field(default="", description="생성된 요구사항 정의서")

class PersonaGenerator:
    """ 페르소나를 생성합니다 """
    def __init__(self, llm: ChatOpenAI, k:int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 사용자 인터뷰용 다양한 페르소나를 만드는 전문가입니다"),
            ("human", f"다음 사용자 요청에 관한 인터뷰를 {self.k}명의 다양한 페르소나를 생성해주세요.\n\n"
                       "사용자 요청: {user_request}\n\n"
                       "각 페르소나에는 이름과 간단한 배경을 포함해주세요. 연령, 성별, 직업, 기술적 전문 지식에서 다양성을 확보해주세요.")
        ])

        # 페르소나 생성을 위한 체인 생성
        chain = prompt | self.llm

        # 페르소나 생성
        return chain.invoke({"user_request": user_request})
    
class InterviewConductor:
    """ 페르소나에게 인터뷰를 실시합니다. """
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        # 질문 생성
        questions = self._generate_questions(user_request=user_request, personas=personas)

        # 답변 생성
        answers = self._generate_answers(personas=personas, questions=questions)

        # 질문과 답변 조합으로 인터뷰 리스트 생성
        interviews = self._create_interviews(personas=personas, questions=questions, answers=answers)

        return InterviewResult(interviews=interviews)

    def _generate_questions(self, user_request: str, personas: list[Persona]) -> list[str]:
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 사용자 요구사항에 기반하여 적절한 질문을 생성하는 전문가입니다."),
            ("human", "다음 페르소나와 관련된 사용자 요청에 대해 하나의 질문을 생성해주세요.\n\n"
                      "사용자 요청: {user_request}\n"
                      "페르소나: {persona_name} - {persona_background}\n\n"
                      "질문은 구체적이며, 이 페르소나의 관점에서 중요한 정보를 끌어낼 수 있도록 설계해주세요.")
        ])

        question_chain = question_prompt | self.llm | StrOutputParser()

        question_queries = [
            {
                "user_request" : user_request,
                "persona_name" : persona.name,
                "persona_background" : persona.background
            } for persona in personas
        ]

        return question_chain.batch(question_queries)

    
    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 다음 페르소나로서 답변하고 있습니다: {persona_name} - {persona_background}"),
            ("human", "질문: {question}")
        ])

        answer_chain = answer_prompt | self.llm | StrOutputParser()

        answer_queries = [
            {
                "persona_name" : persona.name,
                "persona_background" : persona.background,
                "question" : question
            } for persona, question in zip(personas, questions)
        ]

        return answer_chain.batch(answer_queries)

    def _create_interviews(
        self, 
        personas: list[Persona], 
        questions: list[str], 
        answers: list[str]
    ) -> list[Interview]:
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]


class InformationEvaluator:
    """ 수집한 정보의 충분성을 평가합니다. """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(
        self,
        user_request:str,
        interviews: list[Interview]
    )-> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages([
            ("system","당신은 포괄적인 요구사항 문서를 작성하기 위한 정보의 충분성을 평가하는 전문가입니다."),
            ("human", "다음 사용자 요청과 인터뷰 결과를 바탕으로, 포괄적인 요구사항 문서를 작성하기에 충분한 정보가 모였는지 판단해주세요.\n\n"
                      "사용자 요청: {user_request}\n\n"
                      "인터뷰 결과: \n{interview_results}")
        ])

        chain = prompt | self.llm

        return chain.invoke({
            "user_request" : user_request,
            "interview_results" : "\n".join(
                f"페르소나: {i.persona.name} - {i.persona.background}\n"
                f"질문: {i.question}\n답변: {i.answer}\n"
                for i in interviews
            )
        })
    

class RequirementDocumentGenerator:
    """ 요구사항 정의서를 생성합니다. """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 수집한 정보를 바탕으로 요구사항 문서를 작성하는 전문가입니다"),
            ("human", "다음 사용자 요청과 여러 페르소나의 인터뷰 결과를 바탕으로 요구사항 문서를 작성해주세요.\n\n"
                      "사용자 요청: {user_request}\n\n"
                      "인터뷰 결과:\n{interview_results}\n"
                      "요구사항 문서에는 다음 섹션을 포함해주세요:\n"
                      "1. 프로젝트 개요\n"
                      "2. 주요 기능\n"
                      "3. 비기능 요구사항\n"
                      "4. 제약 조건\n"
                      "5. 타깃 사용자\n"
                      "6. 우선순위\n"
                      "7. 위험과 완화 방안\n\n"
                      "출력은 반드시 한국어로 부탁드립니다.\n\n요구사항 문서:")
        ])

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "user_request": user_request,
            "interview_results": "\n".join(
                f"페르소나: {i.persona.name} - {i.persona.background}"
                f"질문: {i.question}\n답변: {i.answer}\n"
                for i in interviews
            )
        })
    
class DocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: int | None):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.information_evaluator = InformationEvaluator(llm=llm)
        self.requirements_generator = RequirementDocumentGenerator(llm=llm)

        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(InterviewState)

        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)

        workflow.set_entry_point("generate_personas")
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {True: "generate_personas", False: "generate_requirements"}
        )

        workflow.add_edge("generate_requirements", END)

        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        new_personas: Personas = self.persona_generator.run(state.user_request)

        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1
        }

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        new_interviews : InterviewResult = self.interview_conductor.run(
            state.user_request, state.personas[-5:]
        )

        return {"interviews" : new_interviews.interviews}

    
    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request, state.interviews
        )

        return {
            "is_information_sufficient" : evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason
        }

    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        requirements_doc : str = self.requirements_generator.run(
            state.user_request, state.interviews
        )

        return {
            "requirements_doc" : requirements_doc
        }

    def run(self, user_request: str) -> str:
        initial_state = InterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state)

        return final_state["requirements_doc"] 
    
def main():
    import argparse

    parser = argparse.ArgumentParser(description="사용자 요구에 기반하여 요구사항 정의를 생성합니다")
    parser.add_argument("--task", type=str, help="만들고 싶은 애플리케이션에 대해 기술해주세요.")
    parser.add_argument("--k", type=int, default=5, help="생성할 페르소나 수를 설정해주세요(기본값: 5)")

    args = parser.parse_args()

    llm = ChatOpenAI(model="gpt-5.4-mini",temperature=0.0)

    agent = DocumentationAgent(llm=llm, k=args.k)
    final_output = agent.run(user_request=args.task)

    print(final_output)

if __name__ == "__main__":
    main()