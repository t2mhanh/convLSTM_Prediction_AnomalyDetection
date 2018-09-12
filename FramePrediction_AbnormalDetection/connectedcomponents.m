% CONNECTEDCOMPONENTS   Label connected components in 2-D binary image.
%   [L,nr,rp] = CONNECTEDCOMPONENTS(I) computes the 8-connected components
%   in the binary image I.
%
%   I must be of type 'logical'.
%
%   L contains the label of the connected component for each pixel.
%   Connected components are numbered by decreasing area. The background
%   (I==false) is labelled zero.
%   nr is the number of connected components.
%   rp contains the area of each connected component.
%
%   Mark Everingham, Leeds 04-Oct-06

function [L,nr,rp] = connectedcomponents(I)

% pad image to avoid boundary checks

[h,w]=size(I);
off=[-h-3 -h-2 -h-1 -1];
I=[false(1,w+2) ; [false(h,1) I false(h,1)] ; false(1,w+2)];

% find components and equivalences

L=ones(h+2,w+2);    % initially label background as 1
e=1;
nr=1;

for xy=find(I)'             % for all 'true' pixels

    l=L(xy+off);            % processed neighbours
    
    if any(l~=1)            

        % >=1 processed neighbours: possibly multiple labels
        
        while any(e(l)~=l)  % resolve equivalence
            l=e(l);
        end

        l0=l(l~=1);         % pick a non-background label
        l0=l0(1);
        L(xy)=l0;           % copy label
        e(l(l~=1))=l0;      % update equivalence
    else
        
        % no processed neighbours: add new label
        
        nr=nr+1;
        L(xy)=nr;
        e(nr)=nr;
    end
end

% resolve equivalences and relabel

while any(e(e)~=e)
    e=e(e);
end
L=e(L);

% renumber contiguous and by decreasing area

c=full(sparse(L,1,1));      % count pixels by label
c(1)=0;                     % ignore background

% Following line requires matlab 7+
%[sc,si]=sort(c,'descend');  % sort by decreasing area

% This works for older versions of matlab (DRM)
c = -1*c ;
[sc,si]=sort(c);  % sort by decreasing area
c = -1*c ;
sc = -1*sc ;

rp=sc(sc>0);                % areas
nr=length(rp);              % number of components

e(si(sc>0))=1:nr;           % assign labels by area
e(1)=0;                     % background always zero
L=e(L(2:end-1,2:end-1));    % relabel and remove padding
    
