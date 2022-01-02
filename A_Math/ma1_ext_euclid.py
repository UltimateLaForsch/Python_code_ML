import pandas as pd

print('Erweiterter Euklid zur Berechnung des ggT')
a = input("Eingabe Zahl 1:  ")
b = input("Eingabe Zahl 2:  ")
a = int(a)
b = int(b)
c = a % b

df = pd.DataFrame(columns=['a', 'b', 'Q', 'k', 'l'])
no_rows = 0
print('\nModulo-Berechnungen zum Euklidischen Algorithmus:')
while c != 0:
    c = a % b
    print(a, ' mod ' , b, ' = ', c)
    q = a//b
    df.loc[no_rows] = [a, b, q, '', '']
    no_rows += 1
    a = b
    b = c
k = 1
l = 0
df.loc[no_rows] = [a, b, '', k, l]
no_rows -= 1
while no_rows >= 0:
    k = df.iloc[no_rows+1, 4]
    l = df.iloc[no_rows+1, 3] - df.iloc[no_rows, 2] * df.iloc[no_rows+1, 4]
    df.iloc[no_rows, 3] = k
    df.iloc[no_rows, 4] = l
    no_rows -= 1
print('\n Erweitertes Euklid-Schema')
print(df.to_string(index=False))

print('\n x = 47x mod 17 = ',(47*16)%17)


