def solve_T_eq(mod, tau):
    z = mod['z'].values
    T = mod['T'].values
    T_out_down = 3500.
    T_out_up = 2*T[-1]-T[-2]
    P = mod['P'].values
    opac= mod['opac'].values
    rho = mod['rho'].values
    fconv = mod['fconv'].values
    cp = mod['cp'].values
    nabla_ad_bottom = mod['nabla_ad'].values[-1]
    length = z.size

    h = 25000.	#grid spacing

    B = 16*sigma*T**3/(3*opac*rho)
    # nize je derivace B pomoci centralnich diferenci
    # krajni body A jsou aproximovany jednostrannymi diferencemi
    A = concatenate( (
        [(B[1]-B[0])/h],
        (B[2:]-B[:-2])/(2*h),
        [(B[-1]-B[-2])/h]) )
    # H...tlakova skala na spodni hranici
    # potrebna pro okrajovou podminku
    H = P[-1]/(rho[-1]*g(z[-1]))

    # definice hlavni diagonaly a vedlejsich diagonal
    d_main = -2*B/h**2-rho*cp/tau
    #d_main[-1] = 1/h
    #d_main[0] = 1

    d_sub = -0.5*A/h+B/h**2
    b_out_down = d_sub[0]
    d_sub = d_sub[1:]
    #d_sub[-1] = -1/h

    d_sup = 0.5*A/h+B/h**2
    b_out_up = d_sup[-1]
    d_sup = d_sup[:-1]
    #d_sup[0] = 0

    # vektor prave strany:
    # derivace F_conv a zbytek
    b = concatenate( (
        #[3500.],
        [(fconv[1]-fconv[0])/h-(rho[0]*cp[0]*T[0]/tau)-b_out_down*T_out_down],
        (fconv[2:]-fconv[:-2])/(2*h)-(rho*cp*T/tau)[1:-1],
        [(fconv[-1]-fconv[-1])/h-(rho[-1]*cp[-1]*T[-1]/tau)-b_out_up*T_out_up] ) )

    data = [d_sub, d_main, d_sup]
    diags = [-1,0,1]

    M = sparse_diags(data, diags, shape=(length, length), format='csc')
    # TODO zkusit teplotu upravit tak, aby v ni nebyl hrbol
    return spsolve(M,b)
