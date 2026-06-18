</> Python

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "0으로 나눌 수 없습니다."
    return a / b

print("간단한 계산기 프로그램")
print("10 + 5 =", add(10, 5))
print("10 - 5 =", subtract(10, 5))
print("10 * 5 =", multiply(10, 5))
print("10 / 5 =", divide(10, 5))
