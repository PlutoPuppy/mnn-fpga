// Cortex-A9 four-query by eight-column FP32 attention microkernels.
// AAPCS: r4-r8 are saved; d0-d7 and d16-d31 are caller-saved.
#if defined(__arm__) && defined(__ARM_NEON)
.syntax unified
.arm
.fpu neon
.text
.align 4
.global TinyVitAttentionQK4x8
.type TinyVitAttentionQK4x8, %function
TinyVitAttentionQK4x8:
    // (four contiguous Q rows, K in feature-major groups of eight, dots, D)
    push {r4-r6, lr}
    add r4, r0, r3, lsl #2
    add r5, r4, r3, lsl #2
    add r6, r5, r3, lsl #2
    veor q8, q8, q8
    veor q9, q9, q9
    veor q10, q10, q10
    veor q11, q11, q11
    veor q12, q12, q12
    veor q13, q13, q13
    veor q14, q14, q14
    veor q15, q15, q15
1:
    vld1.32 {q0, q1}, [r1]!
    vld1.32 {d4[0]}, [r0]!
    vld1.32 {d4[1]}, [r4]!
    vld1.32 {d5[0]}, [r5]!
    vld1.32 {d5[1]}, [r6]!
    vmla.f32 q8, q0, d4[0]
    vmla.f32 q9, q1, d4[0]
    vmla.f32 q10, q0, d4[1]
    vmla.f32 q11, q1, d4[1]
    vmla.f32 q12, q0, d5[0]
    vmla.f32 q13, q1, d5[0]
    vmla.f32 q14, q0, d5[1]
    vmla.f32 q15, q1, d5[1]
    subs r3, r3, #1
    bne 1b
    vst1.32 {q8, q9}, [r2]!
    vst1.32 {q10, q11}, [r2]!
    vst1.32 {q12, q13}, [r2]!
    vst1.32 {q14, q15}, [r2]!
    pop {r4-r6, pc}
.size TinyVitAttentionQK4x8, .-TinyVitAttentionQK4x8

.align 4
.global TinyVitAttentionAV4x8
.type TinyVitAttentionAV4x8, %function
TinyVitAttentionAV4x8:
    // (four probability rows, V+channel, NC4 output, S, D, probability stride, M)
    // Only called for four live query rows and eight live output channels.
    push {r4-r8, lr}
    ldr r7, [sp, #24]
    lsl r7, r7, #2
    ldr r8, [sp, #28]
    lsl r8, r8, #2
    add r4, r0, r8
    add r5, r4, r8
    add r6, r5, r8
    veor q8, q8, q8
    veor q9, q9, q9
    veor q10, q10, q10
    veor q11, q11, q11
    veor q12, q12, q12
    veor q13, q13, q13
    veor q14, q14, q14
    veor q15, q15, q15
2:
    vld1.32 {q0, q1}, [r1]
    add r1, r1, r7
    vld1.32 {d4[0]}, [r0]!
    vld1.32 {d4[1]}, [r4]!
    vld1.32 {d5[0]}, [r5]!
    vld1.32 {d5[1]}, [r6]!
    vmla.f32 q8, q0, d4[0]
    vmla.f32 q9, q1, d4[0]
    vmla.f32 q10, q0, d4[1]
    vmla.f32 q11, q1, d4[1]
    vmla.f32 q12, q0, d5[0]
    vmla.f32 q13, q1, d5[0]
    vmla.f32 q14, q0, d5[1]
    vmla.f32 q15, q1, d5[1]
    subs r3, r3, #1
    bne 2b
    vst1.32 {q8}, [r2]!
    vst1.32 {q10}, [r2]!
    vst1.32 {q12}, [r2]!
    vst1.32 {q14}, [r2]!
    ldr r8, [sp, #32]
    lsl r8, r8, #4
    sub r8, r8, #64
    add r2, r2, r8
    vst1.32 {q9}, [r2]!
    vst1.32 {q11}, [r2]!
    vst1.32 {q13}, [r2]!
    vst1.32 {q15}, [r2]!
    pop {r4-r8, pc}
.size TinyVitAttentionAV4x8, .-TinyVitAttentionAV4x8
#endif
.section .note.GNU-stack,"",%progbits
