from binaryclassification import BinaryClassification


x = [
    [5.1, 4.9, 7.0, 6.4, 5.9, 5.4, 4.6, 5.0, 6.5, 5.7],
    [3.5, 3.0, 3.2, 3.2, 3.0, 3.9, 3.4, 3.4, 2.8, 2.8],
    [1.4, 1.4, 4.7, 4.5, 5.1, 1.7, 1.4, 1.5, 4.6, 4.5]
]
y = [0,0,1,1,1,0,0,0,1,1]
model = BinaryClassification(x,y,epochs = 750)

model.train_the_model(False)

res = model.estimate_value([5.4,3.9,1.7])
print(res)

model.save_the_model("binary1.txt")

model.load_the_model("binary1.txt")
print(model.parameters)