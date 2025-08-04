import requests
import urllib3
import os

# 打印出 requests 库正在使用的代理和证书信息，用于诊断
print(f"HTTP_PROXY env: {os.environ.get('HTTP_PROXY')}")
print(f"HTTPS_PROXY env: {os.environ.get('HTTPS_PROXY')}")
print(f"REQUESTS_CA_BUNDLE env: {os.environ.get('REQUESTS_CA_BUNDLE')}")
print(f"CURL_CA_BUNDLE env: {os.environ.get('CURL_CA_BUNDLE')}")
print("-" * 20)

# --- 测试 1: 正常的 HTTPS 请求 (很可能会失败) ---
try:
    print("【测试1】正在尝试标准连接...")
    response = requests.get("https://huggingface.co", timeout=10)
    print("  ✅ 【测试1】成功! 状态码:", response.status_code)
except Exception as e:
    print(f"  ❌ 【测试1】失败. 错误: {e}")

print("-" * 20)

# --- 测试 2: 禁用 SSL 证书验证的请求 (很可能会成功) ---
# 禁用警告信息
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    print("【测试2】正在尝试禁用 SSL 验证的连接 (verify=False)...")
    response = requests.get("https://huggingface.co", timeout=10, verify=False)
    print("  ✅ 【测试2】成功! 状态码:", response.status_code)
except Exception as e:
    print(f"  ❌ 【测试2】失败. 错误: {e}")