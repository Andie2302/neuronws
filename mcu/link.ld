/* Speicherplan für den emulierten Cortex-M4 (QEMU, Maschine mps2-an386):
   Code und Konstanten ab Adresse 0, Arbeitsspeicher ab 0x20000000.
   Für einen echten Chip werden nur diese beiden Zeilen angepasst, siehe README.md. */
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 1M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 1M
}

ENTRY(reset)

SECTIONS
{
    /* Die Vektortabelle muss ganz vorn liegen. Das erste Wort ist der Startwert des Stack-Zeigers
       (das Ende des Arbeitsspeichers), danach folgt der Einsprung für Reset und die übrigen
       Ausnahmen. */
    .vector_table ORIGIN(FLASH) :
    {
        LONG(ORIGIN(RAM) + LENGTH(RAM));
        KEEP(*(.vector_table.reset_vector));
        KEEP(*(.vector_table.exceptions));
    } > FLASH

    .text : { *(.text .text.*) } > FLASH
    .rodata : { *(.rodata .rodata.*) } > FLASH

    /* Marken, die das Programm zum Messen braucht: Ende des Flash-Inhalts, Anfang des
       Arbeitsspeichers und Ende des Stacks (er wächst von dort nach unten). */
    _flash_end = .;
    _ram_start = ORIGIN(RAM);
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);

    /* Das Programm hat keine veränderlichen Statics: Alles liegt auf dem Stack, und der Startcode
       muss weder `.data` kopieren noch `.bss` nullen. Käme doch etwas dazu, bricht das Linken ab
       (siehe unten), statt dass es zur Laufzeit stillschweigend falsch läuft. */
    .data : { *(.data .data.*) } > RAM
    .bss : { *(.bss .bss.*) *(COMMON) } > RAM

    /DISCARD/ : { *(.ARM.exidx .ARM.exidx.*) *(.ARM.extab .ARM.extab.*) }
}

ASSERT(SIZEOF(.data) + SIZEOF(.bss) == 0, "Das Programm hat veraenderliche Statics (.data oder .bss), der Startcode initialisiert sie nicht")
