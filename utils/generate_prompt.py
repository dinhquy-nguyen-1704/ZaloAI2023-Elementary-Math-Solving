def generate_prompt_train(question, choices, explanation, answer):
    
    answer_option = answer.split('. ')[0]  # Extracting only the answer option letter (e.g., "D")
    
    formatted_answer = f"$\\boxed{{{answer_option}}}$"
    
    return f"""
<|im_start|>system
You are an expert in math. You will receive multiple choice questions with options, solve step by step if available and choose the correct option.

<|im_start|>user
### Question:
{question}
### Choices:
{choices}\nPlease reason step by step, and put your final answer within \\boxed{{}}.
### Answer:

<|im_start|>assistant
{explanation}\nThe answer is {formatted_answer}.
""".strip()


def generate_prompt_test(question, choices):
 
    return f"""
<|im_start|>system
You are an expert in math. You will receive multiple choice questions with options, solve step by step if available and choose the correct option.

<|im_start|>user
### Question:
{question}
### Choices:
{choices}\nPlease reason step by step, and put your final answer within \\boxed{{}}.
### Answer:

<|im_start|>assistant
""".strip()

def generate_few_shot_prompt(question, choices):
 
    return f"""
Below are some examples of tasks and their appropriate responses. After the examples, there is a new task. Write a response that appropriately completes the request.

### Example 1:

<|im_start|>user
### Question:
Chữ số 5 trong số 162,57 thuộc ……………………….
### Choices:
A. Hàng đơn vị và có giá trị $\\\\frac{{5}}{{10}}$ 
B. Hàng phần mười và có giá trị $\\\\frac{{5}}{{10}}$
C. Hàng đơn vị và có giá trị 5
D. D. 26 605 + 8 125
Please reason step by step, and put your final answer within \\boxed{{}}.
### Answer:

<|im_start|>assistant
Trong số 162,57, chữ số 5 đứng sau dấu phẩy, nên nó thuộc hàng phần mười. \n\nVề giá trị của chữ số 5, chúng ta cần xem xét vị trí của nó trong số. Trong trường hợp này, chữ số 5 đứng ở vị trí sau dấu phẩy đầu tiên, nghĩa là nó đại diện cho $\\frac{{5}}{{10}}$ hoặc 0.5, chứ không phải là 5. \n\nVì vậy, câu trả lời chính xác là \"Hàng phần mười và có giá trị $\\frac{{5}}{{10}}$
The answer is $\\boxed{{B}}$.

### Example 2:

<|im_start|>user
### Question:
Một hình bình hành có độ dày đáy bằng $\\frac{{3}}{{2}}$ dm và chiều cao bằng $\\frac{{1}}{{2}}$ độ dài đáy. Diện tích của hình bình hành là:
### Choices:
A. $\\\\frac{{9}}{{4}}$ dm2
B. $\\\\frac{{9}}{{16}}$ dm2
C. $\\\\frac{{9}}{{8}}$ dm2
D. $\\\\frac{{3}}{{4}}$ dm2
Please reason step by step, and put your final answer within \\boxed{{}}.
### Answer:

<|im_start|>assistant
Diện tích hình bình hành bằng độ dài cạnh đáy nhân với chiều cao.\n Chiều cao của hình bình hành là: $\\frac{{3}}{{2}}$ ${{\\times}}$ $\\frac{{1}}{{2}}$ = $\\frac{{3}}{{4}}$ (dm)\n Diện tích hình bình hành đó là: $\\frac{{3}}{{2}}$ ${{\\times}}$ $\\frac{{3}}{{4}}$ = $\\frac{{9}}{{8}}$ (dm2)\n Đáp số: $\\frac{{9}}{{8}}$ dm2.
The answer is $\\boxed{{C}}$.

### New task:

<|im_start|>system
You are an expert in math. You will receive multiple choice questions with options, solve step by step if available and choose the correct option.

<|im_start|>user
### Question:
{question}
### Choices:
{choices}\nPlease reason step by step, and put your final answer within \\boxed{{}}.
### Answer:

<|im_start|>assistant
""".strip()
