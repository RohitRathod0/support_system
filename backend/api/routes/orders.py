from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/orders", tags=["Orders"])


# ─── Mock order database ──────────────────────────────────────────────────────
def _build_orders():
    today = datetime.now()
    return [
        {
            "order_id": "ORD-7492-BL",
            "product_name": "Ninja Pro Blender 1000W",
            "category": "Home Appliances",
            "order_date": (today - timedelta(days=12)).strftime("%Y-%m-%d"),
            "expected_delivery": (today - timedelta(days=5)).strftime("%Y-%m-%d"),
            "status": "Delivered",
            "amount_rupees": 4500,
            "tracking_number": "TRK-BL-483920",
            "payment_method": "UPI",
            "shipping_address": "Mumbai, Maharashtra",
            "courier": "Delhivery",
            "timeline": [
                {"date": (today - timedelta(days=12)).strftime("%d %b"), "event": "Order Placed",       "done": True},
                {"date": (today - timedelta(days=11)).strftime("%d %b"), "event": "Payment Confirmed",  "done": True},
                {"date": (today - timedelta(days=9)).strftime("%d %b"),  "event": "Shipped",            "done": True},
                {"date": (today - timedelta(days=7)).strftime("%d %b"),  "event": "Out for Delivery",   "done": True},
                {"date": (today - timedelta(days=5)).strftime("%d %b"),  "event": "Delivered",          "done": True},
            ],
            "delay_reason": None,
        },
        {
            "order_id": "ORD-3811-TS",
            "product_name": "Premium Cotton T-Shirt (Navy)",
            "category": "Clothing",
            "order_date": (today - timedelta(days=18)).strftime("%Y-%m-%d"),
            "expected_delivery": (today - timedelta(days=12)).strftime("%Y-%m-%d"),
            "status": "Delivered",
            "amount_rupees": 999,
            "tracking_number": "TRK-TS-219043",
            "payment_method": "Credit Card",
            "shipping_address": "Mumbai, Maharashtra",
            "courier": "BlueDart",
            "timeline": [
                {"date": (today - timedelta(days=18)).strftime("%d %b"), "event": "Order Placed",       "done": True},
                {"date": (today - timedelta(days=17)).strftime("%d %b"), "event": "Payment Confirmed",  "done": True},
                {"date": (today - timedelta(days=15)).strftime("%d %b"), "event": "Shipped",            "done": True},
                {"date": (today - timedelta(days=13)).strftime("%d %b"), "event": "Out for Delivery",   "done": True},
                {"date": (today - timedelta(days=12)).strftime("%d %b"), "event": "Delivered",          "done": True},
            ],
            "delay_reason": None,
        },
        {
            "order_id": "ORD-9932-EB",
            "product_name": "Wireless Noise-Canceling Earbuds",
            "category": "Electronics",
            "order_date": (today - timedelta(days=5)).strftime("%Y-%m-%d"),
            "expected_delivery": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "status": "In Transit",
            "amount_rupees": 8999,
            "tracking_number": "TRK-EB-774412",
            "payment_method": "Net Banking",
            "shipping_address": "Mumbai, Maharashtra",
            "courier": "Ekart Logistics",
            "timeline": [
                {"date": (today - timedelta(days=5)).strftime("%d %b"), "event": "Order Placed",                "done": True},
                {"date": (today - timedelta(days=4)).strftime("%d %b"), "event": "Payment Confirmed",           "done": True},
                {"date": (today - timedelta(days=3)).strftime("%d %b"), "event": "Shipped from Warehouse",      "done": True},
                {"date": (today - timedelta(days=1)).strftime("%d %b"), "event": "Arrived at District Hub",     "done": True},
                {"date": (today + timedelta(days=1)).strftime("%d %b"), "event": "Out for Delivery (Expected)", "done": False},
            ],
            "delay_reason": "Heavy rainfall in the Mumbai region caused a 1-day transit delay at the sorting facility.",
        },
    ]


# Build once at module load
_ORDERS = _build_orders()
_TRACKING_MAP = {o["tracking_number"]: o for o in _ORDERS}
_ORDER_ID_MAP  = {o["order_id"]: o for o in _ORDERS}


@router.get("/")
async def get_mock_orders():
    """Returns all mock orders for the customer portal."""
    return {"orders": _ORDERS}


# NOTE: /tracking/{tracking_number} MUST come before /{order_id}
# so FastAPI doesn't match "tracking" as an order_id.
@router.get("/tracking/{tracking_number}")
async def get_tracking(tracking_number: str):
    """
    Simulate a courier partner tracking lookup.
    Returns a human-readable message the AI agent injects into its response.
    """
    order = _TRACKING_MAP.get(tracking_number.upper())
    if not order:
        raise HTTPException(status_code=404, detail=f"Tracking number {tracking_number} not found.")

    today = datetime.now()

    if order["status"] == "Delivered":
        agent_message = (
            f"We just checked with our delivery partner **{order['courier']}**. "
            f"Your order **{order['order_id']}** ({order['product_name']}) was successfully delivered on "
            f"{order['expected_delivery']}. If you haven't received it, please check with your neighbours "
            f"or building security, as our partner sometimes leaves packages with them."
        )
    else:
        delivery_date = (today + timedelta(days=1)).strftime("%d %b %Y")
        reason = order.get("delay_reason", "High shipment volume in your area.")
        agent_message = (
            f"We just spoke with our delivery partner **{order['courier']}** regarding your order "
            f"**{order['order_id']}** ({order['product_name']}). "
            f"Your package has arrived at the **Mumbai District Delivery Centre** and is scheduled to be "
            f"delivered to you **tomorrow ({delivery_date})**. "
            f"We sincerely apologize for the delay \u2014 {reason} "
            f"Rest assured, your package is safe and will reach you soon. "
            f"Your tracking number is **{tracking_number}** and current status is **{order['status']}**."
        )

    return {
        "tracking_number": tracking_number,
        "order_id": order["order_id"],
        "product_name": order["product_name"],
        "courier": order["courier"],
        "status": order["status"],
        "timeline": order["timeline"],
        "expected_delivery": order["expected_delivery"],
        "delay_reason": order.get("delay_reason"),
        "agent_message": agent_message,
    }


@router.get("/{order_id}")
async def get_order(order_id: str):
    """Get a single order by order ID."""
    order = _ORDER_ID_MAP.get(order_id.upper())
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found.")
    return order
