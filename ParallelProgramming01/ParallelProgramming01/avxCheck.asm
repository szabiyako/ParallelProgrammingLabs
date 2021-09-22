.code
isSupportAVX512 proc
	push rbx
	mov eax, 7
	mov ecx, 0

	cpuid

	shr ebx, 16
	and ebx, 1

	mov eax, ebx

	pop rbx
	ret
isSupportAVX512 endp
end