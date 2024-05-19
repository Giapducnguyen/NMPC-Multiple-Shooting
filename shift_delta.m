function [t0, x0, du0] = shift_delta(T, t0, x0, ukm1, duk, f)
st = x0;
con = ukm1 + duk(1,:)';
f_value = f(st,con);
st = st+ (T*f_value);
x0 = full(st);

t0 = t0 + T;
du0 = [duk(2:size(duk,1),:);duk(size(duk,1),:)];
end