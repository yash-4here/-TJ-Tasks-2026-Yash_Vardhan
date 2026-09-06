w1 = 3.5
w2 = 2.0
bias = -3.0

def forward_pass(x1,x2,w1,w2,bias):
    input1= x1*w1
    input2= x2*w2

    weighted_sum = input1 + input2 + bias
    return input1,input2,weighted_sum

def activate(weighted_sum):
    return 1 if weighted_sum>=0 else 0

def simulate(study_hours,attendance):
    print("="*50)
    print(f"INPUT -> Study Hours={study_hours}, Attendance={attendance}")
    print(f"PARAMETERS -> w1={w1} w2={w2} bias={bias}")

    input1,input2,weighted_sum = forward_pass(study_hours,attendance,w1,w2,bias)

    print(f"MULTIPLY -> {study_hours}x{w1} = {round(input1,2)}")
    print(f"MULTIPLY -> {attendance}x{w2} = {round(input2,2)}")
    print(f"SUM -> {round(input1,2)} + {round(input2, 2)} + ({bias}) = {round(weighted_sum,2)}")

    result = activate(weighted_sum)
    decision = "PASS" if result==1 else "FAIL"
    
    print(f"ACTIVATE -> Is {round(weighted_sum,2)} >= 0 ? -> Output:{result}")
    print(f"FINAL DECISION:{decision}")
    print("="*50)
    print()    
    return result

simulate(0.6, 0.4)
simulate(0.9, 0.1)
simulate(0.1, 0.9)
simulate(0.2, 0.3)
simulate(0.8, 0.9)
