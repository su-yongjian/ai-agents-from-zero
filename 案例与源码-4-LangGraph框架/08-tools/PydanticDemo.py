"""
【案例】Pydantic 入门：类型校验、自动转换与 ValidationError

对应教程章节：第 17 章 - Tools 工具调用 → 4、自定义 Tool → 4.3 Pydantic 与参数 schema（Pydantic 简要入门）

知识点速览：
- Pydantic 基于类型注解在「实例化时」做校验与转换：合法则自动转（如 "18"→18），不合法则抛 ValidationError。
- BaseModel 子类的属性即字段；可用 Field(description=..., gt=..., default=...) 等增强描述与约束。
- StrictInt 等严格类型会拒绝自动转换（如 "41" 不会转成 41），仅接受真实 int，便于在工具参数中严格把关。
"""
from pydantic import BaseModel, ValidationError, StrictInt


# 继承 BaseModel：实例化时按类型注解校验，不合规则抛出 ValidationError
class User(BaseModel):
    # id: int  # 普通 int 时，传入 "41" 会被自动转成 41
    id: StrictInt  # 严格整数：不接受字符串等，必须已是 int，否则报错
    name: str
    age: int = 0  # 可选字段，默认 0；传入值会被校验并转换


try:
    # 合法：id=42 为 int，实例化成功
    u = User(id=42, name="z3")
except ValidationError as e:
    print(e)
print(u.id, type(u.id))  # 42 <class 'int'>

print()
print()

# 非法：id="abc" 无法转为整数，StrictInt 拒绝后抛出 ValidationError
try:
    User(id="abc", name="Bob")
except ValidationError as e:
    print(e)
# 输出示例：
# 1 validation error for User
# id
#   value is not a valid integer (type=type_error.integer)
