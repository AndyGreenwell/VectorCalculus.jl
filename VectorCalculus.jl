module VectorCalculus

export grad, divergence, curl

function gradhelper(f)
    s = size(f,1)
    if s == 1
    
        g = zeros(f)
        
    elseif s == 2
    
        f1 = f[1,:] # Extracts the first "row" from all dimensions.
        f2 = f[2,:]
        n = length(f1)
        f21 = reshape(f2-f1, 1, n)
        g = reshape([f21; f21], size(f))
        
    elseif s == 3
    
        f1 = f[1,:]
        f2 = f[2,:]
        f3 = f[3,:]
        n = length(f1)
        f21 = reshape(f2-f1, 1, n)
        f31 = reshape(f3-f1, 1, n)
        f32 = reshape(f3-f2, 1, n)
        g = reshape([f21 ; f31./2 ; f32], size(f))
        
    else
    
        f1 = f[1,:]
        f2 = f[2,:]
        fe = f[end,:]
        fm = f[end-1,:]
        n = length(f1)
        f21 = reshape(f2-f1, 1, n)
        fem = reshape(fe-fm, 1, n)
        g = reshape([f21 ; (f[3:end,:] - f[1:end-2,:])./2 ; fem], size(f))
        
    end
    
    return g
    
end

function gradhelper(f,h,n,k)

     s = size(f,1)
     
     if s == 1
     
         g = zeros(f)
         
     elseif s == 2
     
         hi = [h[2]-h[1] ; h[k] - h[k-1]]
         f1 = f[1,:]
         f2 = f[2,:]
         fk = f[k,:]
         fm = f[k-1,:]
         n = length(f1)
         f21 = reshape(f2-f1, 1, n)
         fkm = reshape(fk-fm, 1, n)
         g = reshape([f21 ; fkm]./hi[:,ones(Int, div(n, k), 1)], size(f))

     elseif s == 3 && k == 3
     
         hi = [h[2]-h[1] ; (h[3:k] - h[1:k-2]) ; h[k] - h[k-1]]
         f1 = f[1,:]
         f2 = f[2,:]
         f3 = f[3,:]
         n = length(f1)
         f21 = reshape(f2-f1, 1, n)
         f31 = reshape(f3-f1, 1, n)
         f32 = reshape(f3-f2, 1, n)
         g = reshape([f21; f31 ; f32] ./ hi[:,ones(Int, div(n, k), 1)], size(f))
     
     else
     
         hi = [h[2]-h[1] ; (h[3:k] - h[1:k-2]) ; h[k] - h[k-1]]
         
         f1 = f[1,:]
         f2 = f[2,:]
         f3 = f[3:k,:]
         f4 = f[1:k-2,:]
         fk = f[k,:]
         fm = f[k-1,:]
         n = length(f1)
         m = size(f3,1)
         f21 = reshape(f2-f1,1,n)
         fkm = reshape(fk-fm,1,n)
         
         if m == 3
         
             f34 = reshape(f3-f4,1,n)
             g = reshape([f21 ; f34 ; fkm] ./ hi[:,ones(Int, div(n, k), 1)], size(f))
             
         else
         
             g = reshape([f21 ; f3 - f4 ; fkm] ./ hi[:,ones(Int, div(n, k), 1)], size(f))
             
         end

     end
     
     return g
     
end

function grad(f)

    nd = ndims(f)
    grads = Array(Array{eltype(f), nd}, nd)

    for i=1:nd
        dims = [2; 1; collect(3:nd)]
        dims[1] = dims[i]
        dims[i] = 2
        grads[i] = ipermutedims(gradhelper(permutedims(f, dims)), dims)
    end

    return grads
end

function grad{T<:Number}(f, h::Vector{T}...)

    hn = length(h)
    nd = ndims(f)
    n = length(f)
    a = size(f)
    
    @assert nd == hn
    
    grads = Array(Array{eltype(f), nd}, nd)

    for i=1:nd
        dims = [2; 1; collect(3:nd)]
        dims[1] = dims[i]
        dims[i] = 2
        hi = h[i]
        k = a[dims[i]]
        if length(hi) == 1
            grads[i] = ipermutedims(gradhelper(permutedims(f, dims)), dims)./hi[1] 
        elseif length(hi) == a[i]
            grads[i] = ipermutedims(gradhelper(permutedims(f, dims), hi, n, k), dims)
        end
    end
    
    return grads
end

function grad{T<:Number}(f, h::T...)

    hn = length(h)
    nd = ndims(f)
    n = length(f)
    a = size(f)
    
    @assert nd == hn
    
    grads = Array(Array{eltype(f), nd}, nd)

    for i=1:nd
        dims = [2; 1; collect(3:nd)]
        dims[1] = dims[i]
        dims[i] = 2
        hi = h[i]
        k = a[dims[i]]
        grads[i] = ipermutedims(subgrad(permutedims(f,dims)), dims)./hi
    end

    return grads
end


function divergence{T<:Number}(X::Vector{T}, Y::Vector{T}, Z::Vector{T}, U::Array{T,3}, V::Array{T,3}, W::Array{T,3})
    
    ux, uy, uz = grad(U, X, Y, Z)
    vx, vy, vz = grad(V, X, Y, Z)
    wx, wy, wz = grad(W, X, Y, Z)
    
    return ux + vy + wz
end

function divergence{T<:Number}(X::Array{T,3}, Y::Array{T,3}, Z::Array{T,3}, U::Array{T,3}, V::Array{T,3}, W::Array{T,3})

    x = X[1, :, 1]
    y = Y[:, 1, 1]
    z = Z[1, 1, :]
    
    ux, uy, uz = grad(U, x, y, z)
    vx, vy, vz = grad(V, x, y, z)
    wx, wy, wz = grad(W, x, y, z)
    
    return ux + vy + wz
end

function divergence{T<:Number}(X::Vector{T}, Y::Vector{T}, U::Matrix{T}, V::Matrix{T})

    ux, uy = grad(U, X, Y)
    vx, vy = grad(V, X, Y)
    
    return ux + vy
end

function divergence{T<:Number}(X::Matrix{T}, Y::Matrix{T}, U::Matrix{T}, V::Matrix{T})

    x = X[1, :]
    y = Y[:, 1]

    ux, uy = grad(U, x, y)
    vx, vy = grad(V, x, y)
    
    return ux + vy
end

function divergence{T<:Number}(U::Array{T,3}, V::Array{T,3}, W::Array{T,3})
    
    ux, uy, uz = grad(U)
    vx, vy, vz = grad(V)
    wx, wy, wz = grad(W)
    
    return ux + vy + wz
end

function divergence{T<:Number}(U::Matrix{T}, V::Matrix{T})
    
    ux, uy = grad(U)
    vx, vy = grad(V)
    
    return ux + vy
end


function curl{T<:Number}(X::Vector{T}, Y::Vector{T}, Z::Vector{T}, U::Array{T,3}, V::Array{T,3}, W::Array{T,3})
    
    ux, uy, uz = grad(U, X, Y, Z)
    vx, vy, vz = grad(V, X, Y, Z)
    wx, wy, wz = grad(W, X, Y, Z)
    
    cx = wy - vz
    cy = uz - wx
    cz = vx - uy
    ca = (cx.*U + cy.*V + cz.*W) ./ (2sqrt(U.^2 + V.^2 + W.^2))
    
    return cx, cy, cz, ca
end

function curl{T<:Number}(X::Array{T,3}, Y::Array{T,3}, Z::Array{T,3}, U::Array{T,3}, V::Array{T,3}, W::Array{T,3})

    x = X[1, :, 1]
    y = Y[:, 1, 1]
    z = Z[1, 1, :]
    
    ux, uy, uz = grad(U, x, y, z)
    vx, vy, vz = grad(V, x, y, z)
    wx, wy, wz = grad(W, x, y, z)
    
    cx = wy - vz
    cy = uz - wx
    cz = vx - uy
    ca = (cx.*U + cy.*V + cz.*W) ./ (2sqrt(U.^2 + V.^2 + W.^2))
    
    return cx, cy, cz, ca
end

function curl{T<:Number}(X::Vector{T}, Y::Vector{T}, U::Matrix{T}, V::Matrix{T})

    ux, uy = grad(U, X, Y)
    vx, vy = grad(V, X, Y)
    
    cz = vx - uy
    ca = cz./2
    
    return cz, ca
end

function curl{T<:Number}(X::Matrix{T}, Y::Matrix{T}, U::Matrix{T}, V::Matrix{T})

    x = X[1, :]
    y = Y[:, 1]

    ux, uy = grad(U, x, y)
    vx, vy = grad(V, x, y)
    
    cz = vx - uy
    ca = cz./2
    
    return cz, ca
end

function curl{T<:Number}(U::Array{T,3}, V::Array{T,3}, W::Array{T,3})
    
    ux, uy, uz = grad(U)
    vx, vy, vz = grad(V)
    wx, wy, wz = grad(W)
    
    cx = wy - vz
    cy = uz - wx
    cz = vx - uy
    ca = (cx.*U + cy.*V + cz.*W) ./ (2sqrt(U.^2 + V.^2 + W.^2))
    
    return cx, cy, cz, ca
end

function curl{T<:Number}(U::Matrix{T}, V::Matrix{T})
    
    ux, uy = grad(U)
    vx, vy = grad(V)
    
    cz = vx - uy
    ca = cz./2
    
    return cz, ca
end

end
