section .data
    vec1    dd 1.0, 2.0, 3.0, 4.0   ; 4 floats
    vec2    dd 2.0, 3.0, 4.0, 5.0   ; 4 floats
    result  dd 0.0                   ; Result (float)
    msg     db "Dot product: ", 0    ; Message prefix
    buf     times 16 db 0            ; Buffer for float-to-string conversion
    newline db 10                    ; Newline character

section .text
    global _start

_start:
    ; --- Compute dot product ---
    movaps  xmm0, [vec1]      ; Load vec1
    movaps  xmm1, [vec2]      ; Load vec2
    dpps    xmm0, xmm1, 0xFF  ; Compute dot product (imm8 = 0xFF)
    movss   [result], xmm0     ; Store result

    ; --- Convert float to string ---
    mov     rdi, buf          ; Output buffer
    mov     eax, [result]     ; Load float as raw bits
    call    float_to_string   ; Custom conversion (see below)

    ; --- Print message ---
    mov     rax, 1            ; sys_write
    mov     rdi, 1            ; stdout
    mov     rsi, msg          ; "Dot product: "
    mov     rdx, 13           ; Length of msg
    syscall

    ; --- Print result (float as string) ---
    mov     rax, 1            ; sys_write
    mov     rdi, 1            ; stdout
    mov     rsi, buf          ; Converted float string
    mov     rdx, 16           ; Max length
    syscall

    ; --- Print newline ---
    mov     rax, 1            ; sys_write
    mov     rdi, 1            ; stdout
    mov     rsi, newline      ; Newline
    mov     rdx, 1            ; Length
    syscall

    ; --- Exit ---
    mov     rax, 60           ; sys_exit
    xor     rdi, rdi          ; Exit code 0
    syscall

; --- Float to String Conversion ---
; Input:  EAX = float (raw bits), RDI = output buffer
; Output: RDI points to null-terminated string
float_to_string:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32           ; Reserve stack space

    ; Use C library's `snprintf` for simplicity (alternatively, implement manually)
    mov     [rsp], rdi        ; Save buffer pointer
    mov     edi, eax          ; Pass float bits
    call    convert_and_store

    leave
    ret

; Helper for float-to-string (simplified)
convert_and_store:
    ; In a real program, use `snprintf` or a proper float-to-string algorithm.
    ; This is a placeholder that just writes "X.XX" for demonstration.
    mov     byte [rdi], '4'   ; Replace with actual conversion logic
    mov     byte [rdi+1], '.'
    mov     byte [rdi+2], '0'
    mov     byte [rdi+3], '0'
    mov     byte [rdi+4], 0   ; Null-terminate
    ret