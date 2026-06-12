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


C:\Users\lwx1468560\.ssh>ssh root@7.221.78.88 -p 31248 -i C:\Users\lwx1468560\.ssh\ziyi -v
OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2
debug1: Connecting to 7.221.78.88 [7.221.78.88] port 31248.
debug1: Connection established.
debug1: identity file C:\\Users\\lwx1468560\\.ssh\\ziyi type 3
debug1: identity file C:\\Users\\lwx1468560\\.ssh\\ziyi-cert type -1
debug1: Local version string SSH-2.0-OpenSSH_for_Windows_9.5
debug1: Remote protocol version 2.0, remote software version OpenSSH_8.9p1 Ubuntu-3ubuntu0.13
debug1: compat_banner: match: OpenSSH_8.9p1 Ubuntu-3ubuntu0.13 pat OpenSSH* compat 0x04000000
debug1: Authenticating to 7.221.78.88:31248 as 'root'
debug1: load_hostkeys: fopen C:\\Users\\lwx1468560/.ssh/known_hosts2: No such file or directory
debug1: load_hostkeys: fopen __PROGRAMDATA__\\ssh/ssh_known_hosts: No such file or directory
debug1: load_hostkeys: fopen __PROGRAMDATA__\\ssh/ssh_known_hosts2: No such file or directory
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: curve25519-sha256
debug1: kex: host key algorithm: ssh-ed25519
debug1: kex: server->client cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: client->server cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: SSH2_MSG_KEX_ECDH_REPLY received
debug1: Server host key: ssh-ed25519 SHA256:VOxphQ64sR6mz2dlF28PMcfEysAzN8/oVT3Lfy86XiM
debug1: load_hostkeys: fopen C:\\Users\\lwx1468560/.ssh/known_hosts2: No such file or directory
debug1: load_hostkeys: fopen __PROGRAMDATA__\\ssh/ssh_known_hosts: No such file or directory
debug1: load_hostkeys: fopen __PROGRAMDATA__\\ssh/ssh_known_hosts2: No such file or directory
debug1: Host '[7.221.78.88]:31248' is known and matches the ED25519 host key.
debug1: Found key in C:\\Users\\lwx1468560/.ssh/known_hosts:4
debug1: ssh_packet_send2_wrapped: resetting send seqnr 3
debug1: rekey out after 134217728 blocks
debug1: SSH2_MSG_NEWKEYS sent
debug1: expecting SSH2_MSG_NEWKEYS
debug1: ssh_packet_read_poll2: resetting read seqnr 3
debug1: SSH2_MSG_NEWKEYS received
debug1: rekey in after 134217728 blocks
debug1: get_agent_identities: ssh_get_authentication_socket: No such file or directory
debug1: Will attempt key: C:\\Users\\lwx1468560\\.ssh\\ziyi ED25519 SHA256:m5b97publFWuCnCa4hSVnii5hu0JTKCYtUJ93hXSrVU explicit
debug1: SSH2_MSG_EXT_INFO received
debug1: kex_input_ext_info: server-sig-algs=<ssh-ed25519,sk-ssh-ed25519@openssh.com,ssh-rsa,rsa-sha2-256,rsa-sha2-512,ssh-dss,ecdsa-sha2-nistp256,ecdsa-sha2-nistp384,ecdsa-sha2-nistp521,sk-ecdsa-sha2-nistp256@openssh.com,webauthn-sk-ecdsa-sha2-nistp256@openssh.com>
debug1: kex_ext_info_check_ver: publickey-hostbound@openssh.com=<0>
debug1: SSH2_MSG_SERVICE_ACCEPT received
debug1: Authentications that can continue: publickey,password
debug1: Next authentication method: publickey
debug1: Offering public key: C:\\Users\\lwx1468560\\.ssh\\ziyi ED25519 SHA256:m5b97publFWuCnCa4hSVnii5hu0JTKCYtUJ93hXSrVU explicit
debug1: Authentications that can continue: publickey,password
debug1: Next authentication method: password
root@7.221.78.88's password:
