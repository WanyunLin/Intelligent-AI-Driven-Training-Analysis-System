import streamlit as st
import wandb
import base64
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from io import BytesIO

st.markdown("""
    <style>
    /* 放大 selectbox 上方標籤的文字 */
    div.stSelectbox label p {
        font-size: 26px !important;
        font-weight: 600 !important;
        color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='margin-bottom: 30px;'>🔍 AI 輔助訓練結果解析系統</h1>", unsafe_allow_html=True)

# 隱藏"press Enter to apply"
hide_submit_text = """
<style>
div[data-testid="InputInstructions"] > span:nth-child(1) {
    visibility: hidden;
}
</style>
"""
st.markdown(hide_submit_text, unsafe_allow_html=True)

# ---------------------- API 設定 dialog -----------------------
@st.dialog("🔐 API 金鑰設定", dismissible=False)
def api_keys_dialog(logged=False):
    openai_key = st.text_input(
        "OpenAI API Key：", type="password", value=st.session_state.get("openai_key", "")
    )
    wandb_key = st.text_input(
        "WandB API Key：", type="password", value=st.session_state.get("wandb_key", "")
    )
    
    if logged == True:
        col1, col2 = st.columns(2, vertical_alignment="bottom")
        button1 = col1.button("Login", width="stretch")
        button2 = col2.button("Cancel", width="stretch")
    else:
        button1 = st.button("Login", width="stretch")

    if button1:
        if not openai_key or not wandb_key:
            st.error("❌ 請輸入完整 API 金鑰！")
        
        elif wandb_key and len(wandb_key.strip()) != 40:
            st.warning(f"⚠️ 目前輸入長度為 {len(wandb_key.strip())}，應為 40 字元。")
            
        else:
            st.session_state.openai_key = openai_key
            st.session_state.wandb_key = wandb_key

            # try:
                # 確認 api keys
            st.session_state.openai_logged = check_openai_key_valid(st.session_state.openai_key)
            st.session_state.wandb_logged = check_wandb_key_valid(st.session_state.wandb_key)

            if st.session_state.openai_logged and st.session_state.wandb_logged:
                st.session_state.api_verified = True
                st.rerun()
            else:
                st.error("❌ 無效的 API 金鑰錯誤，請重新確認！")
                print(st.session_state.openai_logged, st.session_state.wandb_logged)

    if logged == True:
        if button2:
            st.rerun()

# ------------- Get WanDB Project & Run data ------------------

def get_projects(api, entity):
    projects = api.projects(entity)
    return [p.name for p in projects]

def get_runs(api, entity, project_name):
    runs = api.runs(f"{entity}/{project_name}")
    return {r.displayName: r.id for r in runs}

def get_run_object(api, entity, project_name, run_id):
    return api.run(f"{entity}/{project_name}/{run_id}")

# ----------------- [UI] WanDB Project & Run -----------------

def get_project_run_selection(api, entity):
    project_names = get_projects(api, entity)
    runs_zip = []

    for p in project_names:
        runs_zip.append(get_runs(api, entity, p))

    return project_names, runs_zip


def validate_selection(project, run):
    if not project or not run:
        st.warning("⚠️ 請先選擇完整的 Project 與 Run 再執行！")
        return False
    return True

# ------------- WanDB get traning datasets -----------------

def plot_wandb_history(history, target_cols, group_size=6):
    figures = []

    for i in range(0, len(target_cols), group_size):
        subset = target_cols[i:i + group_size]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for j, col in enumerate(subset):
            ax = axes[j]
            ax.plot(history.index, history[col], label=col, linewidth=0.8)
            ax.set_xlabel("Step")
            ax.set_ylabel(col.split("/")[1])
            ax.set_title(col.split("/")[1])

        # 關閉多餘的子圖
        for k in range(len(subset), len(axes)):
            axes[k].axis("off")

        plt.tight_layout()
        figures.append(fig)

    return figures

# ------------------------- GPT 問答 ------------------------------

def analyze_plot_with_gpt(image_base64, client, user_query=None):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # 第一次: 自動分析訊息
    if user_query is None:
        user_msg = {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "請分析這張 Reward 圖表的趨勢，說明訓練過程中模型的學習狀況與可能的問題。"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    else:
        # 後續提問：一般純文字訊息
        user_msg = {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": user_query},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }

    st.session_state.chat_history.append(user_msg)

    try:
        with st.spinner("GPT 正在思考..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.chat_history
            )

        assistant_reply = response.choices[0].message.content

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_reply
        })

        return assistant_reply
    
    except Exception as e:
        st.error("⚠️ GPT 分析失敗，請檢查 API 或網路連線")
        return None
    
def display_chat(user_prompt, gpt_answer):
    if user_prompt != None:
        # Display user message in chat message container
        st.chat_message("user").markdown(user_prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

    if gpt_answer != None:
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(gpt_answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": gpt_answer})


def check_openai_key_valid(api_key):
    try:
        client = OpenAI(api_key=api_key)
        _ = client.models.list()  # 輕量 API 請求
        return True
    except Exception as e:
        print("OpenAI Key 驗證失敗：", e)
        return False

def check_wandb_key_valid(api_key):
    try:
        wandb.login(key=api_key)
        _ = wandb.Api()
        return True
    except Exception as e:
        print("WanDB Key 驗證失敗：", e)
        return False


# --------------------- Main -------------------------
def main():
    # ===== streamlit 參數設定 =====
    ## 狀態初始化
    if "api_verified" not in st.session_state:
        st.session_state.api_verified = False
    if "selectbox_modified" not in st.session_state:
        st.session_state.selectbox_modified = False

    ## API keys
    if "wandb_logged" not in st.session_state:
        st.session_state.wandb_logged = False
    if "openai_logged" not in st.session_state:
        st.session_state.openai_logged = False
    if "api" not in st.session_state:
        st.session_state.api = None
    if "client" not in st.session_state:
        st.session_state.client = None

    ## 圖片&對話
    if "project_list" not in st.session_state:
        st.session_state.project_list = []
    if "runs_list" not in st.session_state:
        st.session_state.runs_list = []
    if "Graph_display" not in st.session_state:
        st.session_state.Graph_display = None    # AF:不需要這樣寫
    if "Graph" not in st.session_state:
        st.session_state.Graph = None    # AF:不需要這樣寫
    if "first_quest" not in st.session_state:
        st.session_state.first_quest = True

    # Initialize chat history
    if "messages" not in st.session_state:
      st.session_state.messages = []


    # === 固定區塊 ===
    top_slot = st.container()      # 上方：選單 + 圖表


    # ===== dialog =====
    if not st.session_state.api_verified:
        api_keys_dialog()
        st.stop()

    if st.sidebar.button("⚙️ 設定 API"):
        api_keys_dialog(logged=True)
        st.stop()
    

    # ===== 確認 api keys =====
    ## OPENAI
    # st.session_state.openai_logged = check_openai_key_valid(st.session_state.openai_key)
    client = OpenAI(api_key=st.session_state.openai_key)

    ## WanDB
    # st.session_state.wandb_logged = check_wandb_key_valid(st.session_state.wandb_key)
    wandb.login(key=st.session_state.wandb_key)
    api = wandb.Api()
    entity = api.default_entity

    if len(st.session_state.project_list) == 0:
        st.session_state.project_list, st.session_state.runs_list = get_project_run_selection(api, entity)


    with top_slot:
        # ===== 下拉式選單 =====
        col1, col2 = st.columns(2)

        with col1:
            selected_project = st.selectbox("📁 選擇專案 (Project)：", st.session_state.project_list)

        with col2:
            if selected_project:
                runs = st.session_state.runs_list[st.session_state.project_list.index(selected_project)]
                selected_run = st.selectbox("🧪 選擇實驗 (Run)：", runs.keys())
                if len(runs)>0:
                    selected_run = runs[selected_run]
            else:
                selected_run = None

        button_showGrpah = st.button("🚀 Show Graph")
        
        divide = st.empty()    # 分隔線
        Graph = st.empty()
        if st.session_state.Graph_display is not None:
            divide.divider()    # 分隔線
            Graph.pyplot(st.session_state.Graph_display, width="content")


        # ===== 從WanDB資料產生圖片
        if button_showGrpah:
            if validate_selection(selected_project, selected_run):
                run = get_run_object(api, entity, selected_project, selected_run)
                # st.success(f"✅ 已成功載入 Run：{run.name}")
                divide.divider()    # 分隔線
                
                # 製圖
                with st.spinner("根據 Reward Data 生成折線圖..."):
                    history = run.history(samples=10000)
                    target_cols = [
                        col for col in history.columns
                        if pd.api.types.is_numeric_dtype(history[col]) and col.split("/")[0] == "Episode_Reward"
                    ]
                    figures = plot_wandb_history(history, target_cols)

                # 存圖再讀
                fig = figures[0]
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                # 顯示圖表
                st.session_state.Graph_display = fig
                Graph.pyplot(st.session_state.Graph_display, width="content")
                # st.chat_message("user").markdown(fig)   # 不能這樣用

                st.session_state.Graph = image_base64
                plt.close(fig)

            else:
                st.session_state.Graph = None
                st.stop()
    

    st.divider()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 第一次自動詢問 GPT 並顯示回覆
    if st.session_state.first_quest and st.session_state.Graph is not None:
        gpt_answer = analyze_plot_with_gpt(st.session_state.Graph, client)
        with st.chat_message("assistant"):
            st.markdown(gpt_answer)
        st.session_state.messages.append({"role": "assistant", "content": gpt_answer})
        st.session_state.first_quest = False

    # 後續由使用者輸入提問
    if not st.session_state.first_quest:
        if user_prompt := st.chat_input("根據這張圖其他問題？"):
            gpt_answer = analyze_plot_with_gpt(st.session_state.Graph, client, user_prompt)
            with st.chat_message("user"):
                st.markdown(user_prompt)
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("assistant"):
                st.markdown(gpt_answer)
            st.session_state.messages.append({"role": "assistant", "content": gpt_answer})

if __name__ == "__main__":
    main()
