import json
import logging
import subprocess
import os
from typing import Dict, Any
from google import genai
from google.genai import types

# Tool functions for code generation and execution
def execute_command(command: str) -> str:
    """Execute a shell command and return its output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

def create_file(path: str, content: str) -> str:
    """Create a new file with the given content"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully created file: {path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

def read_file(path: str) -> str:
    """Read content of a file"""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Tool declarations
tool_declarations = [
    {
        "name": "execute_command",
        "description": "Execute a shell command on the system",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "create_file",
        "description": "Create a new file with specified content",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path where the file should be created",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read content of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to read",
                },
            },
            "required": ["path"],
        },
    },
]

SYSTEM_PROMPT = """
You are an AI coding assistant that helps users build and modify code projects.
You work in [START], [THINK], [ACTION], [OBSERVE], and [OUTPUT] Mode.

In the [START] phase, user gives a coding task or query.
Then, you [THINK] about how to implement the solution, considering:
1. Required files and their structure
2. Dependencies and setup needed
3. Implementation approach
4. Testing strategy

If you need to execute commands or create/modify files, use the available tools:
- execute_command: Run shell commands
- create_file: Create new files with content
- read_file: Read existing file contents

[Rules]:
- Always think through the solution before taking action
- Execute one step at a time and observe results
- Verify file contents and command outputs
- Provide clear explanations of your actions
- Follow best practices for the target language/framework

Output Format:
{ "step": "string", "tool": "string", "input": "python_dict", "content": "string" }
"""

def talk_to_ai():
    client = genai.Client(
        vertexai=True,
        project="sounish-cloud-workstation",
        location="global",
    )

    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=1,
        seed=0,
        max_output_tokens=1000,
        system_instruction=SYSTEM_PROMPT,
        tool_config=types.Tool(function_declarations=tool_declarations),
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
    )

    print("Welcome! I'm your AI coding assistant. What would you like to build?")
    user_query = input("user: ")
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=user_query)],
        ),
    ]

    while True:
        model = "gemini-2.0-flash"
        
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        output = resp.candidates[0].content.parts[0].text
        functool = resp.candidates[0].content.parts[0].function_call

        logging.info("function tool: %s", functool)

        if functool:
            if functool.name == "execute_command":
                result = execute_command(**functool.args)
            elif functool.name == "create_file":
                result = create_file(**functool.args)
            elif functool.name == "read_file":
                result = read_file(**functool.args)
            
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=result)],
                ),
            )

        logging.info("AI: %s", output)
        if "[OUTPUT]" in output:
            break
        contents.append(
            types.Content(
                role="model",
                parts=[types.Part(text=output)],
            ),
        )

def main():
    talk_to_ai()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    main()


# build a simple todo-app in a folder name "todo-app-proj" in HTML, CSS, JS. make it fully functional and working. use Funky color scheme and fonts
