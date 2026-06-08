from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.database import get_db
from backend.app.dependencies.auth_dependency import get_current_user
from backend.app.models.user import User
from backend.app.schemas.auth_schema import (
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from backend.app.services.auth_service import (
    create_access_token,
    hash_password,
    verify_password,
)


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post("/register", response_model=UserResponse)
def register_user(
    request: UserRegisterRequest,
    db: Session = Depends(get_db),
) -> User:
    existing_user = db.query(User).filter(User.email == request.email).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    new_user = User(
        name=request.name,
        email=request.email,
        password_hash=hash_password(request.password),
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@router.post("/login", response_model=TokenResponse)
def login_user(
    request: UserLoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    user = db.query(User).filter(User.email == request.email).first()

    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
        }
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=user,
    )


@router.get("/me", response_model=UserResponse)
def get_logged_in_user(
    current_user: User = Depends(get_current_user),
) -> User:
    return current_user