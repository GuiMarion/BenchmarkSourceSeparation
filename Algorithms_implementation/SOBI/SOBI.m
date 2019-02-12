function [H,S] = SOBI(X,n,p)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                        
    % Based on the Flexible Audio Source Separatqion Toolbox (FASST), Version 1.0 
    %
    %  A. Ozerov, E. Vincent and F. Bimbot                                                              
    % "A General Flexible Framework for the Handling of Prior Information in Audio Source Separation," 
    % IEEE Transactions on Audio, Speech and Signal Processing 20(4), pp. 1118-1133 (2012).                                                   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
    % Note: > m: sensor number.
    %       > n: sources number (by default m=n).
    %       > p: number of correlation matrices to be diagonalized (by default p=4).

    % Retrieve the number of sensors (m) and the number of samples (N).
    [m,N]=size(X);
    
    % Default values: number of sources (n) and correlation matrices (p).
    if nargin==1
        p=4;
        n=m;
    elseif nargin==2
        p=4;
    end
    
    % Impose zero-mean mixtures.
    X=X-kron(mean(X')',ones(1,N));

    % Whitening step based on SVD.
    [~,S,VV]=svd(X',0);
    Q= pinv(S)*VV';
    X=Q*X;
    
    % Estimation of correlation matrices.
    k=1;
    pm = p*m;
    for i=1:m:pm
        k=k+1; 
        Rxp=X(:,k:N)*X(:,1:N-k+1)'/(N-k+1); 
        M(:,i:i+m-1)=norm(Rxp,'fro')*Rxp; 
    end;
    
    % SOBI alogrithm aims at retrieving the most relevant rotation U that
    % can be used to retrieve the sources.
    % This is done thanks to a joint diagonalization of the correlation
    % matrices by using Givens rotations.
    epsilon=1/sqrt(N)/100; 
    keep_going=1; 
    V=eye(m);
    while keep_going
        keep_going=0;
        for p=1:m-1
            for q=p+1:m
               % Givens rotations
               g=[    M(p,p:m:pm)-M(q,q:m:pm)  ;
                      M(p,q:m:pm)+M(q,p:m:pm)  ;
                  1i*(M(q,p:m:pm)-M(p,q:m:pm))];
              
              [vcp,D] = eig(real(g*g')); 
              [~,K] = sort(diag(D));
              angles = vcp(:,K(3));
              angles = sign(angles(1))*angles;
              c = sqrt(0.5+angles(1)/2);
              sr = 0.5*(angles(2)-1i*angles(3))/c; 
              sc = conj(sr);
              ok = abs(sr)>epsilon;
              keep_going = keep_going | ok;
              
              % Update the M and V matrices 
              if ok
                colp=M(:,p:m:pm);
                colq=M(:,q:m:pm);
                M(:,p:m:pm)=c*colp+sr*colq;
                M(:,q:m:pm)=c*colq-sc*colp;
                rowp=M(p,:);
                rowq=M(q,:);
                M(p,:)=c*rowp+sc*rowq;
                M(q,:)=c*rowq-sr*rowp;
                temp=V(:,p);
                V(:,p)=c*V(:,p)+sr*V(:,q);
                V(:,q)=c*V(:,q)-sc*temp;
              end %if-loop
            end %q for-loop
        end %p for-loop
    end %while-loop

    % Estimation of the mixing matrix and source signals
    Hf=pinv(Q)*V; % estimated mixing matrix
    Sf=V'*X; % estimated sources

    H = Hf(:,1:n);
    S = Sf(1:n,:);

end