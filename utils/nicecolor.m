function c = nicecolor(s)
if strcmp(s,'red')==1
    c_array = [ 228, 26, 28 ];   % red
elseif strcmp(s,'blue')==1
    c_array = [ 55, 126, 184 ];  % blue
elseif strcmp(s,'green')==1
    c_array = [ 77, 175, 74 ];   % green
elseif strcmp(s,'purple')==1
    c_array = [ 152, 78, 163 ];  % purple
elseif strcmp(s,'orange')==1
    c_array = [ 255, 127, 0 ];   % orange
elseif strcmp(s,'yellow')==1
    c_array = [ 255, 255, 51 ];  % yellow
elseif strcmp(s,'brown')==1
    c_array = [ 166, 86, 40 ];   % brown
elseif strcmp(s,'pink')==1
    c_array = [ 247, 129, 191 ]; % pink
elseif strcmp(s,'grey')==1
    c_array = [ 153, 153, 153];  % grey
elseif strcmp(s,'black')==1
    c_array = [ 0, 0, 0];  % black
end
c = c_array./255;
end