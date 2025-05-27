import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import os

# 初始化模型变量
gpt4_model = None

# 定义创建文章多智能体团队的函数
def 创建文章团队(主题):
    # 角色信息字典，方便循环创建
    角色信息 = {
        "研究员": {
            "目标": "对主题进行深入调研，收集关键信息、数据和专家观点",
            "背景": "你是一名细致入微的专业研究员",
        },
        "撰稿人": {
            "目标": "根据调研内容撰写详细且生动的文章，使用Markdown格式",
            "背景": "你是一名擅长写作和排版的内容专家",
        },
        "编辑": {
            "目标": "审核并优化文章，确保内容清晰、准确且格式规范",
            "背景": "你是一名经验丰富的编辑，注重内容质量和排版",
        }
    }

    # 创建Agent对象列表
    agents = []
    for 角色, 信息 in 角色信息.items():
        agents.append(Agent(
            role=角色,
            goal=信息["目标"],
            backstory=信息["背景"],
            verbose=True,
            allow_delegation=False,
            llm=gpt4_model
        ))

    # 创建任务描述字典，顺序对应agents列表
    任务描述 = [
        f"请围绕主题“{主题}”进行全面调研，收集关键数据和专家观点。",
        """请根据调研内容撰写一篇结构清晰、内容丰富的文章，要求：
        - 使用Markdown格式，包含主标题（H1）、章节标题（H2）、小节标题（H3）
        - 适当使用项目符号或编号列表
        - 重点内容加粗或斜体强调
        - 文章条理清晰，易于阅读""",
        """请审核文章，确保：
        - Markdown格式正确且统一
        - 标题层级合理
        - 内容逻辑流畅，吸引读者
        - 重点突出
        并对内容和格式进行必要的修改和提升。"""
    ]

    预期输出 = [
        "调研报告，包含关键数据和专家观点。",
        "基于调研的详细且格式规范的Markdown文章。",
        "最终润色后的高质量文章。"
    ]

    # 创建Task对象列表
    tasks = []
    for i in range(len(agents)):
        tasks.append(Task(
            description=任务描述[i],
            agent=agents[i],
            expected_output=预期输出[i]
        ))

    # 创建团队，顺序执行任务
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=2,
        process=Process.sequential
    )
    return crew

# Streamlit 页面配置和样式
st.set_page_config(page_title="多智能体AI写作助手", page_icon="📝")

st.markdown("""
<style>
.stApp {
    max-width: 1800px;
    margin: 0 auto;
    font-family: "微软雅黑", Arial, sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
.stTextInput>div>div>input {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.title("📝 多智能体AI写作助手")

# 侧边栏输入API Key
with st.sidebar:
    st.header("配置")
    api_key = st.text_input("请输入OpenAI API密钥（API Key）：", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        gpt4_model = ChatOpenAI(model_name="gpt-4o-mini")
        st.success("API密钥设置成功！")
    else:
        st.info("请输入OpenAI API密钥以继续。")

st.markdown("使用AI多智能体，快速生成高质量文章！")

subject = st.text_input("请输入文章主题：", placeholder="例如：人工智能对医疗行业的影响")

if st.button("生成文章"):
    if not api_key:
        st.error("请在侧边栏输入OpenAI API密钥。")
    elif not subject.strip():
        st.warning("请输入文章主题。")
    else:
        with st.spinner("🤖 AI智能体正在生成文章，请稍候..."):
            团队 = 创建文章团队(subject)
            结果 = 团队.kickoff()
            st.markdown(结果)

st.markdown("---")
st.markdown("由CrewAI和OpenAI强力驱动 ❤️")
