function koh_pred = koh(train_xl,train_yl,train_xh,train_yh,test_x)
cell_x{1} = train_xl; cell_x{2} = train_xh;
cell_y{1} = train_yl; cell_y{2} = train_yh;
        
koh = oodacefit(cell_x',cell_y');
koh_pred = koh.predict(test_x);   % KOH prediction results
end