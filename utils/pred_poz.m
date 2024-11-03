function pred_poz = pred_poz(lb1,ub1,lb2,ub2,N)
ss1 = gridsamp([lb1;ub1],N);
ss2 = gridsamp([lb2;ub2],N);
p=1;
for i = 1:N
    for j = 1:N
        pred_poz(p,1) = ss1(i);
        pred_poz(p,2) = ss2(j);
        p=p+1;
    end
end
end
