# 								NLP HW1

###																Manxueying Li, UNI:ml4529



**<u>Analytical Component:</u>**

## Problem1

### (i)

$P(spam) = \frac{3}{5}\\P(ham) = \frac{2}{5}$

### (ii)

<img src="/Users/Ene/Library/Mobile Documents/com~apple~CloudDocs/CU/2020Fall/nlp/hw1/Q1_table.png" alt="Q1_table"  />

### (iii)

$\begin{align*}y_1 &= argmax_y P(y)\prod_i P(x_i|y)\\ &= \begin{cases} y_{spam} = P(Nigeria|Spam)P(Spam)= \frac{1}{10}\\y_{ham} =P(Nigeria|Ham)P(Ham))=\frac{2}{35}\end{cases}\\&= spam \end{align*}$

Therefore, predicted label for “Nigeria” is Spam.



$\begin{align*}y_2 &= argmax_y P(y)\prod_i P(x_i|y)\\ &= \begin{cases} y_{spam} = P(Spam)P(Nigeria|Spam)P(home|Spam)= \frac{1}{10}\cdot \frac{1}{12} = 0.00833\\y_{ham} =P(Ham)P(Nigeria|Ham)P(home|Ham)=\frac{2}{35}\cdot \frac{2}{7}=0.016327)\end{cases}\\&= ham \end{align*}$

Therefore, predicted label for “Nigeria hom”e is Ham.



$\begin{align*}y_3 &= argmax_y P(y)\prod_i P(x_i|y)\\ &= \begin{cases} y_{spam} = P(Spam)P(home|Spam)P(bank|Spam)P(money|Spam)= \frac{3}{5}\cdot \frac{1}{12}\cdot \frac{2}{12}\cdot \frac{1}{12} = 0.000694\\y_{ham} =P(Ham)P(home|Ham)P(bank|Ham)P(money|Ham)=\frac{2}{5}\cdot \frac{1}{7}\cdot \frac{2}{7}\cdot \frac{1}{7} = 0.002332\end{cases}\\&= ham \end{align*}$

Therefore, predicted label for “home bank money” is Ham.



## Problem2

$\begin{align*}\sum_{w_1,w_2,…,w_n} P(w_1,w_2,…,w_n)&=\sum_{w_1,w_2,…,w_n} P(w_1|start)P(w_2|w_1)P(w_3,w_2)P(w_4,w_5)…P(w_n|w_{n-1})\\&=\sum_{w_n} P(w_n|w_{n-1})\sum_{w_1,w_2,…,w_{n-1}} P(w_1|start)P(w_2|w_1)P(w_3,w_2)P(w_4,w_5)…P(w_{n-1}|w_{n-2}) \text{Summing over all possibility of $w_n$}\\&=1\cdot \sum_{w_1,w_2,…,w_{n-1}} P(w_1|start)P(w_2|w_1)P(w_3,w_2)P(w_4,w_5)…P(w_{n-1}|w_{n-2}) \text{Marginalize $w_n$}\\&=\sum_{w_{n-1}} P(w_{n-1}|w_{n-2})\sum_{w_1,w_2,…,w_{n-2}} P(w_1|start)P(w_2|w_1)P(w_3,w_2)P(w_4,w_5)…P(w_{n-2}|w_{n-3}) \text{Summing over all possibility of $w_{n-1}$}\\&=1\cdot \sum_{w_1,w_2,…,w_{n-2}} P(w_1|start)P(w_2|w_1)P(w_3,w_2)P(w_4,w_5)…P(w_{n-1}|w_{n-3}) \text{Marginalize $w_{n-1}$}\\\\ &\text{do the same marginaliztion for the rest $w_i \in \{w_1,w_2,…,w_{n-2}\}$. Since we sum over all possibility of every w, every term will become 1}\\\\&= 1\end{align*}$



**<u>Programming Component:</u>**

