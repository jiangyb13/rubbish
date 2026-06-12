You are an Image Improvement Assistant. Your job is
to help make the image more aligned with the ORIGINAL
prompt.
Given Inputs:
1. Original User Prompt:
• {original prompt}
2. History of Prompt Refinements and Feedback:
• {history text}
3. Current Image Analysis:
• Look at the image and identify what aspects DIFFER
from what the ORIGINAL prompt requested.
• Analyze what essential elements from the ORIGINAL
prompt are missing or incorrectly represented.
Your Task:
1. Create a NEW PROMPT that will help generate an image
that better matches the ORIGINAL prompt.
2. Focus on fixing what’s missing or incorrectly
represented in the current image
3. Incorporate suitable elements from prompt history
into the NEW PROMPT.
4. The goal is to get progressively closer to
fulfilling the ORIGINAL prompt.
Output:
REFINED PROMPT:
"<A detailed and enhanced version of the last prompt
that improves alignment>“

Please carefully examine this generated image and compare it with the original
prompt:
"{original_prompt}"
Analyze the following aspects:
1. Does the image accurately represent the main subject described in the
prompt?
2. Are the visual details (clothing, environment, style, etc.) consistent with
the prompt?
3. Is the overall mood and atmosphere matching the intended description?
4. Are there any missing elements or incorrect interpretations?
If the image matches the prompt well, respond with: "MATCH: The image
successfully represents the prompt."
If there are discrepancies, respond with: "EDIT_NEEDED: [specific editing
instructions]"
For example: "EDIT_NEEDED: The character should be wearing a red dress
instead of blue, and the background should be a forest not a city."
Please be specific about what needs to be changed:


import re

def parse_model_feedback(feedback_text):
    """
    鲁棒地解析模型的审查反馈，无视 Markdown 符号和前置废话。
    返回一个 tuple: (status, instruction)
    status 可以是 "MATCH", "EDIT_NEEDED", 或 "UNKNOWN"
    """
    # 1. 过滤掉可能干扰的 Markdown 加粗/斜体符号
    clean_text = feedback_text.replace('*', '').strip()
    
    # 2. 优先匹配 EDIT_NEEDED 并提取后面的指令
    # re.IGNORECASE: 忽略大小写
    # re.DOTALL: 允许提取跨越多行的指令
    edit_match = re.search(r'EDIT_NEEDED:\s*(.*)', clean_text, re.IGNORECASE | re.DOTALL)
    if edit_match:
        # 提取出具体的修改建议
        instruction = edit_match.group(1).strip()
        return "EDIT_NEEDED", instruction
        
    # 3. 匹配 MATCH 关键字
    # 只要文本中独立出现了 MATCH（忽略大小写），就认为通过
    match_found = re.search(r'\bMATCH\b', clean_text, re.IGNORECASE)
    if match_found:
        return "MATCH", None
        
    # 4. 如果模型完全胡言乱语，没有按指令输出
    return "UNKNOWN", clean_text
