import streamlit as st
import replicate
import time
import requests
import zipfile
import io
from PIL import Image
from replicate.exceptions import ReplicateError

# --- 页面配置 ---
st.set_page_config(page_title="二次元转3D工作台", layout="wide")
st.title("🖥️ 二次元转 3D 游戏质感工作台 (最终稳定版)")
st.markdown("ℹ️ **说明**：已内置模型版本号，无需联网查询，速度更快。请确保 Token 正确。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("🔑 密钥设置")
    raw_token = st.text_input("Replicate API Token", type="password", help="r8_开头")
    api_token = raw_token.strip() if raw_token else None

    if api_token and not api_token.startswith("r8_"):
        st.error("❌ Token 格式错误！必须以 r8_ 开头")

    st.divider()
    
    st.header("🎮 3D化参数")
    condition_scale = st.slider("线稿锁死程度", 0.0, 1.0, 0.55, help="推荐 0.55。过高脸会假，过低脸会变。")
    use_3d_prompt = st.checkbox("✅ 强制 3D 增强咒语", value=True)

# --- 核心工具函数 ---
def preprocess_image(file_obj):
    try:
        image = Image.open(file_obj).convert('RGB')
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        return buf
    except Exception as e:
        raise Exception(f"图片清洗失败: {e}")

def run_replicate_direct(model_version_id, input_data, token):
    """
    直接使用版本 ID 调用，不再查询模型，速度更快，更稳定。
    """
    if not token:
        raise Exception("Token 未填写")
        
    client = replicate.Client(api_token=token)
    
    for attempt in range(3):
        try:
            # 直接创建预测，不再 create(version=...)
            # 这里的 model_version_id 是长字符串 hash
            prediction = client.predictions.create(version=model_version_id, input=input_data)
            prediction.wait()
            
            if prediction.status == "succeeded":
                return prediction.output
            elif prediction.status == "failed":
                err = str(prediction.error)
                if "NSFW" in err: raise Exception("NSFW_ERROR")
                raise Exception(f"API报错: {err}")
                
        except ReplicateError as e:
            # 专门捕捉 401 错误
            if "401" in str(e) or "Unauthenticated" in str(e):
                raise Exception("⛔ 认证失败：Token 无效或已过期！请去 Replicate 重新生成。")
            
            if "429" in str(e):
                st.toast(f"⏳ 限流冷却中... {10 + attempt * 5}s")
                time.sleep(10 + attempt * 5)
                continue
            raise e
        except Exception as e:
            raise e

    raise Exception("连接超时，请检查网络")

def download_image(url):
    return requests.get(url).content

# --- 主界面 ---
left, right = st.columns([1, 1.5], gap="large")

with left:
    st.header("1️⃣ 上传原图")
    ref_file = st.file_uploader("上传二次元图片", type=['jpg', 'png'], key="ref")
    if ref_file:
        st.image(ref_file, caption="原图", use_container_width=True)

with right:
    st.header("2️⃣ 3D 转换结果")
    
    if ref_file and api_token:
        if st.button("🚀 立即转换"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1. 识别内容
                status_text.info("👁️ 正在识别画面内容...")
                clean_img = preprocess_image(ref_file)
                
                # BLIP 模型版本 ID (硬编码，永不过期)
                blip_version = "2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746"
                content_desc = run_replicate_direct(
                    blip_version,
                    {"image": clean_img, "task": "image_captioning"}, 
                    api_token
                )
                content_clean = content_desc.replace("cartoon", "").replace("anime", "").replace("drawing", "").strip()
                
                # 2. 构建提示词
                if use_3d_prompt:
                    final_prompt = (
                        f"Hyper-realistic 3d render of {content_clean}, "
                        "Unreal Engine 5 style, cinematic lighting, 8k resolution, "
                        "highly detailed human skin texture, realistic eyes, ray tracing, "
                        "depth of field, masterpiece, CGI, photograph"
                    )
                else:
                    final_prompt = f"{content_clean}, 3d render, best quality"

                negative_prompt = "anime, cartoon, 2d, sketch, drawing, illustration, painting, flat color, cel shading, vector art"

                # 3. 生成 3D 图
                status_text.info("🎨 正在进行 3D 渲染 (ControlNet)...")
                
                # ControlNet Canny 模型版本 ID (硬编码)
                # 对应 fofr/sdxl-controlnet-canny
                canny_version = "af1a68a271597604546c09c64a844d1502ad61958b9f71c4961501700685608d"
                
                output = run_replicate_direct(
                    canny_version,
                    {
                        "image": clean_img,
                        "prompt": final_prompt,
                        "negative_prompt": negative_prompt,
                        "condition_scale": condition_scale,
                        "num_inference_steps": 50,
                        "guidance_scale": 7.5
                    },
                    api_token
                )
                
                img_url = output[0] if isinstance(output, list) else output
                
                st.image(img_url, caption="3D 结果", use_container_width=True)
                st.markdown(f"**提示词:** `{final_prompt}`")
                st.markdown(f"[下载大图]({img_url})")
                
                status_text.success("✅ 完成！")
                progress_bar.progress(1.0)
                
            except Exception as e:
                # 错误信息会非常直观
                st.error(f"❌ 发生错误: {str(e)}")
                if "Token" in str(e):
                    st.warning("👉 请检查左侧 Token 是否填写正确，或者是否包含空格。")

    elif not api_token:
        st.warning("👈 请输入 Token")
