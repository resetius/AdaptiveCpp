# Bug: dynamic_function test fails on macOS

## Корневая причина

На macOS Clang генерирует два конструктора по Itanium ABI: `C1` (полный, обёртка) и `C2` (базовый, с реальным телом). При `[[clang::noinline]]` на `dynamic_function_definition` конструкторе:

- Аннотации (`dynamic_function_def_arg0`, `needs_function_ptr_argument_reflection`) оказываются только в `C2`
- `C1` — тонкая обёртка вокруг `C2`, которая при O0 сохраняет аргументы в alloca и передаёт их в `C2` через load

## Как должно работать (Linux)

На Linux Clang генерирует **один** конструктор:

```
main()
  │
  ├─── dynamic_function_definition(&myfunction1)
  │         │
  │         └─── C1_constructor(this, &myfunction1)   ← аннотация здесь
  │                   │
  │                   └─── resolve_function_name(&myfunction1)  ✓
  │
  └─── dynamic_function_definition(&myfunction2)
            │
            └─── C1_constructor(this, &myfunction2)   ← аннотация здесь
                      │
                      └─── resolve_function_name(&myfunction2)  ✓
```

Аннотация `dynamic_function_def_arg0` живёт в `C1`. Компилятор смотрит на пользователей `C1` — видит `&myfunction1` и `&myfunction2` напрямую. Всё ок.

## Что происходит на macOS (Itanium ABI C1/C2 split)

На macOS Clang создаёт **два** конструктора:

```
main()
  │
  ├─── C1_constructor(this, &myfunction1)     ← прямой указатель ✓
  │         │
  │         │  [O0: сохраняет arg в alloca, грузит обратно]
  │         │
  │         └─── C2_constructor(this, %load)  ← аннотация ЗДЕСЬ
  │                                  ↑
  │                              не Function*!
  │                              это LoadInst
  │
  └─── C1_constructor(this, &myfunction2)     ← прямой указатель ✓
            │
            └─── C2_constructor(this, %load)  ← аннотация ЗДЕСЬ
```

### Проблема 1: компиляция падает

Компилятор-пасс ищет, кто вызывает аннотированную функцию (`C2`), и смотрит на аргумент:

```
Пасс: "покажи мне аргумент[1] всех вызовов C2"

C1 → C2(%load)    ← getOperand(1) = %load = LoadInst  ≠ llvm::Function*
                                                            └── ОШИБКА!
```

### Проблема 2: runtime падает

`FunctionNameExtractionPass` регистрирует указатели на функции в таблице имён. Он спрашивает: *"есть ли среди прямых callers этой функции аннотированный?"*

```
myfunction1:
  кто вызывает с &myfunction1?
    → define(cfg, &myfunction1, ...)     ← аннотирован ✓  → регистрирует
    → C1_constructor(this, &myfunction1) ← НЕ аннотирован

myfunction2:
  кто вызывает с &myfunction2?
    → C1_constructor(this, &myfunction2) ← НЕ аннотирован
                                             └── не регистрирует!

Runtime: resolve_function_name(&myfunction2) → null → exception 💥
```

## Фиксы

### Фикс 1: `src/compiler/sscp/DynamicFunctionSupport.cpp`

Вместо `dyn_cast<Function>(getOperand(1))` — рекурсивно прослеживаем цепочку:

```
getOperand(1) = %load
                  │
                  └── load из alloca
                            │
                            └── store %arg1 → alloca
                                      │
                                      └── %arg1 = аргумент C1
                                                    │
                                        смотрим на callers C1:
                                          C1(this, &myfunction1) → Function* ✓
                                          C1(this, &myfunction2) → Function* ✓
                                        → собираем оба имени
```

Добавлена функция `collectFunctions()`, которая:
1. Прямой `llvm::Function*` — возвращает сразу
2. `LoadInst` из alloca — смотрит все `StoreInst` в ту alloca
3. `Argument` — смотрит на всех callers функции

### Фикс 2: `src/compiler/reflection/FunctionNameExtractionPass.cpp`

Расширена проверка в `isAnyUserReflectionAnnotatedFunction`: *"аннотирован ли callee, **или** вызывает ли он аннотированного?"*

```
myfunction2:
  → C1_constructor(this, &myfunction2)
        C1 аннотирован? НЕТ
        C1 вызывает аннотированного?
          C1 → C2  ← C2 аннотирован! ДА ✓
        → регистрирует myfunction2 ✓
```

Добавлена вспомогательная функция `callsReflectionAnnotatedFunction()`.
