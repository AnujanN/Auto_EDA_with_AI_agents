from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from config import config
from tools import (
    load_data_tool,
    missing_value_analysis_tool,
    univariate_analysis_tool,
    correlation_analysis_tool,
    outlier_detection_tool,
    target_relationship_tool,
    generate_visualizations_tool,
    export_report_tool
)

@CrewBase
class EDACrew:
    """EDA crew for automated exploratory data analysis"""

    agents_config = config.agents_path
    tasks_config = config.tasks_path

    @agent
    def EDAAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["EDAAgent"],
            tools=[
                load_data_tool,
                missing_value_analysis_tool,
                univariate_analysis_tool,
                correlation_analysis_tool,
                outlier_detection_tool,
                target_relationship_tool,
                generate_visualizations_tool,
                export_report_tool
            ],
            llm=config.llm,
            memory=True,
        )
    
    @agent
    def StreamlitAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["StreamlitAgent"],
            llm=config.llm,
            memory=True,
        )

    @task
    def EDATask(self) -> Task:
        return Task(
            config=self.tasks_config["EDATask"],
            agent=self.EDAAgent(),
            tools=[
                load_data_tool,
                missing_value_analysis_tool,
                univariate_analysis_tool,
                correlation_analysis_tool,
                outlier_detection_tool,
                target_relationship_tool,
                generate_visualizations_tool,
                export_report_tool
            ],
        )
    
    @task
    def StreamlitFormatTask(self) -> Task:
        return Task(
            config=self.tasks_config["StreamlitFormatTask"],
            agent=self.StreamlitAgent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
        )
