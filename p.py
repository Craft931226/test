def lcs(text1, text2):
    m = len(text1)
    n = len(text2)
    c = [[None]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if text1[j] == text2[i]:
                c[i+1][j+1] = c[i][j] +1
            else:
                c[i+1][j+1] = max(c[i][j+1],c[i+1][j])
    print(c[n][m])