def find_s(training_data):
  hypothesis = ['⌀'] * (len(training_data[0])-1)

  for ex in training_data:
    if ex[-1] == 'Yes':
      for i in range (len(hypothesis)):
        if hypothesis[i] == '⌀':
          hypothesis[i]=ex[i]

        elif hypothesis[i]!=ex[i]:
          hypothesis[i] = '?'

  return hypothesis

training_data = [
    ['Sunny','Warm','Normal', 'Strong', 'Yes'],
    ['Sunny','Warm','High', 'Strong', 'Yes'],
    ['Rainy','Cold','High', 'Strong', 'No'],
    ['Sunny','Warm','Normal', 'Weak', 'Yes'],
]

ans = find_s(training_data)
print("Final Hypothesis: ", ans)
