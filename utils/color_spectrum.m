function col = color_spectrum(p)
    no_col = [1 1 1];
    full_col = [1 0 0];
    col = (1 - p)*no_col + p*full_col;
end