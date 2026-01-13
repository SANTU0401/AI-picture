import streamlit as st
import replicate
import time

# 页面配置
st.set_page_config(page_title="AI批量风格迁移工具", layout="wide")

st.title("🎨 AI图片风格提取与批量生成工具")

# --- 侧边栏：设置 ---
with st.sidebar:
    st.header("🔑 设置")
    # 增加 .strip() 说明，防止用户不知道空格问题
    raw_token = st.text_input("输入 Replicate API Token", type="password", help="以 r8_ 开头")
    # 强制清理 Token：去除前后空格、换行符
    api_token = raw_token.strip() if raw_token else None
    
    if api_token and not api_token.startswith("r8_"):
        st.error("⚠️ Token 格式看起来不对，必须以 r8_ 开头")

    st.header("⚙️ 生成参数")
    strength = st.slider("风格重绘幅度 (Strength)", 0.1, 1.0, 0.75, help="推荐 0.7-0.8")
    num_steps = st.slider("生成步数", 20, 50, 30)

# --- 核心函数：封装调用过程，强制传递Token ---
def run_replicate(model, input_data, token):
    try:
        # 显式创建一个客户端，确保使用的是用户输入的Token
        client = replicate.Client(api_token=token)
        return client.run(model, input=input_data)
    except Exception as e:
        raise e

# --- 步骤 1: 上传与分析 ---
st.subheader("1. 上传参考风格图")
ref_file = st.file_uploader("上传参考图", type=['png', 'jpg', 'jpeg'])

if ref_file and api_token:
    st.image(ref_file, caption="参考图", width=250)
    
    if st.button("🔍 分析风格提示词"):
        if not api_token.startswith("r8_"):
            st.error("请先在左侧填入正确的 Token (r8_开头)")
        else:
            with st.spinner("AI正在读取图片..."):
                try:
                    # 使用强力修正后的调用方式
                    output = run_replicate(
                        "pharmapsychotic/clip-interrogator:a24998d0ddb2eabd20197e9e38ef2049d59e99dd94ca9e87900408cb837130b0",
                        {"image": ref_file, "mode": "fast"},
                        api_token
                    )
                    st.session_state['style_prompt'] = output
                    st.success("✅ 提取成功！")
                except Exception as e:
                    st.error(f"分析失败，错误详情: {str(e)}")

# 显示提示词
if 'style_prompt' in st.session_state:
    st.markdown("### 📝 风格提示词")
    style_prompt = st.text_area("提示词", st.session_state['style_prompt'], height=80)
else:
    style_prompt = ""

st.markdown("---")

# --- 步骤 2: 批量处理 ---
st.subheader("2. 批量生成")
uploaded_files = st.file_uploader("上传内容图 (支持多选)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files and style_prompt and api_token:
    if st.button(f"🚀 开始生成 ({len(uploaded_files)} 张)"):
        
        progress_bar = st.progress(0)
        results_area = st.container()
        
        for idx, img_file in enumerate(uploaded_files):
            with st.spinner(f"正在生成第 {idx+1} 张..."):
                try:
                    final_prompt = f"{style_prompt}, high quality, 4k"
                    
                    # 使用强力修正后的调用方式
                    output = run_replicate(
                        "stability-ai/sdxl:39ed52f2a78e934b3ba6e399ea1a963986eeac40ef080b697b0803a6466b717c",
                        {
                            "image": img_file,
                            "prompt": final_prompt,
                            "prompt_strength": 1.0 - strength,
                            "num_inference_steps": num_steps
                        },
                        api_token
                    )
                    
                    # 展示结果
                    with results_area:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img_file, caption="原图", width=200)
                        with col2:
                            st.image(output[0], caption="AI生成图", width=200)
                            st.markdown(f"[下载大图]({output[0]})")
                        st.markdown("---")
                        
                except Exception as e:
                    st.error(f"图片 {img_file.name} 失败: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        st.success("全部完成！")

elif not api_token:
    st.warning("👈 请先在左侧输入 Token")
