export class EventBus {
    constructor() {
        // Хранилище, где ключи - названия событий, а значения - массивы функций
        this.events = {}; 
    }

    // Метод подписки на событие
    on(eventName, callback) {
        if (!this.events[eventName]) {
            this.events[eventName] = [];
        }
        this.events[eventName].push(callback);
    }

    // Метод вызова события с передачей данных (payload)
    emit(eventName, payload) {
        if (this.events[eventName]) {
            // Запускаем все функции, подписанные на это событие
            this.events[eventName].forEach(callback => callback(payload));
        }
    }
}

// Экспортируем единый экземпляр для всего проекта
export const eventBus = new EventBus();