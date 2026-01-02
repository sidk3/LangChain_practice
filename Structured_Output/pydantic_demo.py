from pydantic import BaseModel,Field
from typing import Optional
class Student(BaseModel):
    name:str='Kronos'
    age:Optional[int]=None
    cgpa:float=Field(gt=0,lt=10)
new_stu={'cgpa':8.8}
stu=Student(**new_stu)

print(stu)