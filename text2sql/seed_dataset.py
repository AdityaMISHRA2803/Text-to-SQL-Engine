"""Seed the development database with a realistic e-commerce schema and data.

Tables: categories, customers, products, orders, order_items, payments.

Run as a module:
    python -m src.seed_dataset
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

from sqlalchemy import Numeric, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

try:
    from faker import Faker
except ImportError:  # pragma: no cover - Faker is a required dependency
    Faker = None  # type: ignore

from text2sql.config import get_settings


SEED = 42


class Base(DeclarativeBase):
    pass


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(unique=True, nullable=False)


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column(nullable=False)
    last_name: Mapped[str] = mapped_column(nullable=False)
    country: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False, default="")
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    category_id: Mapped[int] = mapped_column(nullable=False)
    stock: Mapped[int] = mapped_column(nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(nullable=False)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(nullable=False)
    total: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    shipped_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(nullable=False)
    product_id: Mapped[int] = mapped_column(nullable=False)
    quantity: Mapped[int] = mapped_column(nullable=False)
    unit_price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    method: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)


CATEGORIES: list[tuple[str, str]] = [
    ("Electronics", "electronics"),
    ("Books", "books"),
    ("Clothing", "clothing"),
    ("Home & Kitchen", "home-kitchen"),
    ("Sports & Outdoors", "sports-outdoors"),
]

PRODUCT_NAMES: dict[str, list[str]] = {
    "electronics": [
        "Wireless Headphones",
        "Bluetooth Speaker",
        "Smart Watch",
        "USB-C Charger",
        "Portable SSD",
    ],
    "books": [
        "The Pragmatic Programmer",
        "Designing Data-Intensive Applications",
        "Clean Code",
        "Atomic Habits",
        "Deep Work",
    ],
    "clothing": [
        "Classic T-Shirt",
        "Denim Jeans",
        "Hoodie",
        "Running Shorts",
        "Wool Socks",
    ],
    "home-kitchen": [
        "Espresso Machine",
        "Cast Iron Skillet",
        "Stand Mixer",
        "Knife Set",
        "Air Fryer",
    ],
    "sports-outdoors": [
        "Yoga Mat",
        "Camping Tent",
        "Running Shoes",
        "Climbing Rope",
        "Cycling Helmet",
    ],
}

ORDER_STATUSES = ["pending", "paid", "shipped", "delivered", "cancelled"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer"]
PAYMENT_STATUSES = ["pending", "completed", "refunded", "failed"]

COUNTRIES = ["USA", "Canada", "UK", "Germany", "France", "Australia", "Japan", "Brazil"]


def _resolve_engine(engine_or_url: Union[Engine, str, None]) -> Engine:
    if engine_or_url is None:
        return create_engine(get_settings().database_url, future=True)
    if isinstance(engine_or_url, Engine):
        return engine_or_url
    return create_engine(engine_or_url, future=True)


def seed(engine_or_url: Union[Engine, str, None] = None, row_count: int = 100) -> dict[str, int]:
    """Drop + recreate all tables and insert ``row_count`` rows per table."""
    engine = _resolve_engine(engine_or_url)

    random.seed(SEED)
    if Faker is not None:
        Faker.seed(SEED)
        faker: Optional["Faker"] = Faker()
    else:  # pragma: no cover
        faker = None

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine, future=True)
    with SessionLocal() as session:
        # Categories
        category_rows = [Category(name=name, slug=slug) for name, slug in CATEGORIES]
        session.add_all(category_rows)
        session.flush()
        category_ids = [c.id for c in category_rows]

        # Customers
        customers: list[Customer] = []
        now = datetime.now(timezone.utc)
        for i in range(row_count):
            if faker is not None:
                first = faker.first_name()
                last = faker.last_name()
                email = faker.unique.email()
                country = random.choice(COUNTRIES)
            else:  # pragma: no cover
                first, last = f"First{i}", f"Last{i}"
                email = f"user{i}@example.com"
                country = COUNTRIES[i % len(COUNTRIES)]
            customers.append(
                Customer(
                    email=email,
                    first_name=first,
                    last_name=last,
                    country=country,
                    created_at=now - timedelta(days=random.randint(0, 365)),
                )
            )
        session.add_all(customers)
        session.flush()

        # Products
        products: list[Product] = []
        for _ in range(row_count):
            cat_idx = random.randrange(len(category_ids))
            cat_name = CATEGORIES[cat_idx][0]
            cat_key = CATEGORIES[cat_idx][1]
            base_name = random.choice(PRODUCT_NAMES[cat_key])
            price = round(random.uniform(9.99, 999.99), 2)
            products.append(
                Product(
                    name=base_name,
                    description=f"A high-quality {base_name.lower()} from the {cat_name} collection.",
                    price=price,
                    category_id=category_ids[cat_idx],
                    stock=random.randint(0, 500),
                    created_at=now - timedelta(days=random.randint(0, 365)),
                )
            )
        session.add_all(products)
        session.flush()

        # Orders
        orders: list[Order] = []
        for _ in range(row_count):
            cust = random.choice(customers)
            status = random.choice(ORDER_STATUSES)
            shipped_at: Optional[datetime] = None
            if status in {"shipped", "delivered"}:
                shipped_at = now - timedelta(days=random.randint(0, 30))
            orders.append(
                Order(
                    customer_id=cust.id,
                    status=status,
                    total=0,
                    created_at=cust.created_at + timedelta(days=random.randint(0, 200)),
                    shipped_at=shipped_at,
                )
            )
        session.add_all(orders)
        session.flush()

        # Order items + compute order totals
        order_items: list[OrderItem] = []
        for order in orders:
            num_items = random.randint(1, 4)
            chosen = random.sample(products, k=min(num_items, len(products)))
            order_total = 0.0
            for prod in chosen:
                qty = random.randint(1, 5)
                unit_price = float(prod.price)
                order_total += qty * unit_price
                order_items.append(
                    OrderItem(
                        order_id=order.id,
                        product_id=prod.id,
                        quantity=qty,
                        unit_price=unit_price,
                    )
                )
            order.total = round(order_total, 2)
        session.add_all(order_items)

        # Payments
        payments: list[Payment] = []
        for order in orders:
            if order.status == "cancelled":
                pay_status = "refunded"
            elif order.status in {"paid", "shipped", "delivered"}:
                pay_status = "completed"
            else:
                pay_status = random.choice(["pending", "failed"])
            payments.append(
                Payment(
                    order_id=order.id,
                    amount=order.total,
                    method=random.choice(PAYMENT_METHODS),
                    status=pay_status,
                    created_at=order.created_at + timedelta(minutes=random.randint(1, 120)),
                )
            )
        session.add_all(payments)
        session.commit()

        counts = {
            "categories": session.scalar(select(Category).order_by(Category.id.desc())) is not None
            and session.scalar(select(Category.id).order_by(Category.id.desc())) or 0,
        }
        counts = {
            "categories": session.scalar(select(Category.id).order_by(Category.id.desc())) or 0,
            "customers": session.scalar(select(Customer.id).order_by(Customer.id.desc())) or 0,
            "products": session.scalar(select(Product.id).order_by(Product.id.desc())) or 0,
            "orders": session.scalar(select(Order.id).order_by(Order.id.desc())) or 0,
            "order_items": session.scalar(select(OrderItem.id).order_by(OrderItem.id.desc())) or 0,
            "payments": session.scalar(select(Payment.id).order_by(Payment.id.desc())) or 0,
        }

    return counts


def main() -> int:
    counts = seed()
    print("Seed complete. Row counts (max id per table):")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
